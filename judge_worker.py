#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def result_payload(
    *,
    success: bool,
    status: str,
    message: str,
    tests: list[dict[str, Any]] | None = None,
    output: str = "",
    started_at: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tests = tests or []
    summary = {
        "passed": sum(1 for item in tests if item.get("status") == "passed"),
        "failed": sum(1 for item in tests if item.get("status") == "failed"),
        "skipped": sum(1 for item in tests if item.get("status") == "skipped"),
    }
    payload = {
        "success": success,
        "status": status,
        "message": message,
        "summary": summary,
        "tests": tests,
        "durationMs": int((time.perf_counter() - (started_at or time.perf_counter())) * 1000),
        "output": truncate(output),
    }
    if extra:
        payload.update(extra)
    return payload


def apply_resource_limits() -> None:
    cpu_seconds = int(os.environ.get("LEETGPU_CPU_SECONDS", "0"))
    if cpu_seconds > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))

    memory_mb = int(os.environ.get("LEETGPU_MEMORY_MB", "0"))
    if memory_mb > 0:
        bytes_limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def pick_device(torch, requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_torch_for_correctness(torch) -> None:
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def dtype_from_name(torch, dtype: Any):
    if not isinstance(dtype, str):
        return dtype
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "double": torch.float64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "int32": torch.int32,
        "uint32": getattr(torch, "uint32", torch.int32),
        "int64": torch.int64,
        "long": torch.int64,
        "bool": torch.bool,
    }
    if dtype not in mapping:
        raise RuntimeError(f"Unsupported tensor dtype descriptor: {dtype}")
    return mapping[dtype]


def materialize_value(value: Any, torch, device: str) -> Any:
    class_name = value.__class__.__name__
    if class_name == "RandTensor":
        return torch.empty(value.shape, device=device, dtype=dtype_from_name(torch, value.dtype)).uniform_(
            value.low, value.high
        )
    if class_name == "RandnTensor":
        return torch.empty(value.shape, device=device, dtype=dtype_from_name(torch, value.dtype)).normal_(
            value.mean, value.std
        )
    if class_name == "RandIntTensor":
        return torch.randint(
            value.low,
            value.high,
            value.shape,
            device=device,
            dtype=dtype_from_name(torch, value.dtype),
        )
    if class_name == "FullTensor":
        return torch.full(
            value.shape,
            value.value,
            device=device,
            dtype=dtype_from_name(torch, value.dtype),
        )
    if class_name == "OutTensor":
        return torch.empty(value.shape, device=device, dtype=dtype_from_name(torch, value.dtype))
    return value


def materialize_case(case: dict[str, Any], torch, device: str) -> dict[str, Any]:
    return {key: materialize_value(value, torch, device) for key, value in case.items()}


def clone_value(value: Any, torch, device: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device)
    try:
        import torch.nn as nn

        if isinstance(value, nn.Module):
            return copy.deepcopy(value).to(device)
    except Exception:
        pass
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def clone_case(case: dict[str, Any], torch, device: str) -> dict[str, Any]:
    return {key: clone_value(value, torch, device) for key, value in case.items()}


def value_bytes(value: Any, torch) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.nelement() * value.element_size())
    try:
        import torch.nn as nn

        if isinstance(value, nn.Module):
            total = 0
            for tensor in list(value.parameters()) + list(value.buffers()):
                total += int(tensor.nelement() * tensor.element_size())
            return total
    except Exception:
        pass
    return 0


def case_bytes(case: dict[str, Any], torch) -> int:
    return sum(value_bytes(value, torch) for value in case.values())


def normalize_cases(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    raise RuntimeError("Test generator returned neither a dict nor a list of dicts")


def output_keys_from_signature(signature: dict[str, Any]) -> list[str]:
    keys = []
    for key, descriptor in signature.items():
        direction = descriptor[1] if isinstance(descriptor, tuple) and len(descriptor) > 1 else None
        if direction in {"out", "inout"}:
            keys.append(key)
    return keys


def args_for_case(signature: dict[str, Any], case: dict[str, Any]) -> list[Any]:
    ordered_names = [name for name in signature.keys() if name in case]
    if ordered_names:
        return [case[name] for name in ordered_names]
    return list(case.values())


def synchronize(torch, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def return_overrides(return_value: Any, output_keys: list[str], torch) -> dict[str, Any]:
    if return_value is None or not output_keys:
        return {}
    if isinstance(return_value, torch.Tensor):
        return {output_keys[0]: return_value}
    if isinstance(return_value, (tuple, list)):
        return {
            key: value
            for key, value in zip(output_keys, return_value)
            if isinstance(value, torch.Tensor)
        }
    return {}


def tensor_preview(tensor, torch) -> str:
    with torch.no_grad():
        flat = tensor.detach().reshape(-1)
        sample = flat[: min(6, flat.numel())].cpu().tolist()
    return str(sample)


def direction_for_key(signature: dict[str, Any], key: str) -> str | None:
    descriptor = signature.get(key)
    if isinstance(descriptor, tuple) and len(descriptor) > 1:
        return descriptor[1]
    return None


def summarize_value(value: Any, torch) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "numel": int(value.numel()),
            "preview": tensor_preview(value, torch),
        }

    try:
        import torch.nn as nn

        if isinstance(value, nn.Module):
            parameter_count = sum(parameter.numel() for parameter in value.parameters())
            return {
                "kind": "module",
                "type": value.__class__.__name__,
                "parameters": int(parameter_count),
            }
    except Exception:
        pass

    return {
        "kind": "value",
        "value": truncate(repr(value), 240),
    }


def summarize_case(case: dict[str, Any], signature: dict[str, Any], torch) -> list[dict[str, Any]]:
    summaries = []
    for key, value in case.items():
        direction = direction_for_key(signature, key)
        if direction == "out":
            continue
        summaries.append({"name": key, "direction": direction or "value", **summarize_value(value, torch)})
    return summaries


def summarize_output_buffers(
    case: dict[str, Any],
    output_keys: list[str],
    torch,
    *,
    include_preview: bool,
) -> list[dict[str, Any]]:
    summaries = []
    for key in output_keys:
        value = case.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        summaries.append(
            {
                "name": key,
                "shape": list(value.shape),
                "dtype": str(value.dtype).replace("torch.", ""),
                "preview": tensor_preview(value, torch) if include_preview else "",
            }
        )
    return summaries


def output_buffer_description(case: dict[str, Any], output_keys: list[str], torch) -> str:
    summaries = summarize_output_buffers(case, output_keys, torch, include_preview=False)
    return ", ".join(
        f"{item['name']} {item['dtype']} {item['shape']}" for item in summaries
    )


def logical_2d_shape_for_output(case: dict[str, Any], output_key: str, torch) -> tuple[int, int] | None:
    output = case.get(output_key)
    if not isinstance(output, torch.Tensor) or output.dim() != 1:
        return None

    matrix_rows = case.get("M")
    matrix_cols = case.get("N")
    if isinstance(matrix_rows, int) and isinstance(matrix_cols, int):
        if matrix_rows > 0 and matrix_cols > 0 and output.numel() == matrix_rows * matrix_cols:
            return matrix_rows, matrix_cols

    input_rows = case.get("input_rows")
    input_cols = case.get("input_cols")
    if not isinstance(input_rows, int) or not isinstance(input_cols, int):
        return None

    candidates = [(input_rows, input_cols)]
    kernel_rows = case.get("kernel_rows")
    kernel_cols = case.get("kernel_cols")
    if isinstance(kernel_rows, int) and isinstance(kernel_cols, int):
        candidates.append((input_rows - kernel_rows + 1, input_cols - kernel_cols + 1))

    for rows, cols in candidates:
        if rows > 0 and cols > 0 and output.numel() == rows * cols:
            return rows, cols
    return None


def solution_exception_hint(
    exception: Exception,
    case: dict[str, Any],
    output_case: dict[str, Any],
    output_keys: list[str],
    torch,
) -> str:
    message = str(exception)
    lower_message = message.lower()
    hints = []
    output_description = output_buffer_description(output_case, output_keys, torch)

    shape_related = (
        "expand(" in message
        or "shape mismatch" in lower_message
        or "size mismatch" in lower_message
        or "number of sizes provided" in lower_message
        or "must match" in lower_message
    )
    if shape_related and output_description:
        hints.append(f"Expected output buffer shape: {output_description}.")

    if shape_related:
        context_case = {**case, **output_case}
        for key in output_keys:
            logical_shape = logical_2d_shape_for_output(context_case, key, torch)
            if logical_shape:
                rows, cols = logical_shape
                hints.append(
                    f"{key} is a flat row-major buffer for a logical {rows} x {cols} result. "
                    f"If you computed a 2D tensor, write {key}[:] = value.reshape(-1) "
                    f"or {key}.copy_(value.reshape_as({key}))."
                )
                break

    unsupported_int_matmul = (
        "not implemented" in lower_message
        and (
            "addmm_cuda" in lower_message
            or "mm_cuda" in lower_message
            or "bmm_cuda" in lower_message
        )
        and ("'int'" in lower_message or "torch.int32" in lower_message)
    )
    if unsupported_int_matmul:
        hints.append(
            "PyTorch CUDA does not implement int32 matrix multiplication for @/torch.matmul. "
            "Use a supported PyTorch formulation such as "
            "(A[:, :, None] * B[None, :, :]).sum(dim=1), or cast to float32 if that is acceptable "
            "for the challenge."
        )

    return "\n".join(hints)


def compare_tensors(actual, expected, atol: float, rtol: float, torch) -> tuple[bool, str]:
    if not isinstance(actual, torch.Tensor):
        return False, f"actual value is {type(actual).__name__}, expected a tensor"
    same_shape = tuple(actual.shape) == tuple(expected.shape)
    singleton_shape_compatible = actual.numel() == expected.numel() == 1
    if not same_shape and not singleton_shape_compatible:
        return False, f"shape mismatch: got {tuple(actual.shape)}, expected {tuple(expected.shape)}"

    if actual.dtype != expected.dtype:
        try:
            actual_for_compare = actual.to(expected.dtype)
        except Exception:
            return False, f"dtype mismatch: got {actual.dtype}, expected {expected.dtype}"
    else:
        actual_for_compare = actual

    if not same_shape:
        actual_for_compare = actual_for_compare.reshape(expected.shape)

    if expected.dtype.is_floating_point or expected.dtype.is_complex:
        close = torch.isclose(actual_for_compare, expected, rtol=rtol, atol=atol, equal_nan=True)
        if bool(torch.all(close).item()):
            return True, ""
        mismatch = torch.nonzero(~close.reshape(-1), as_tuple=False)
        index = int(mismatch[0].item()) if mismatch.numel() else 0
        actual_value = actual_for_compare.detach().reshape(-1)[index].cpu().item()
        expected_value = expected.detach().reshape(-1)[index].cpu().item()
        max_error = torch.max(torch.abs(actual_for_compare - expected)).detach().cpu().item()
        return (
            False,
            f"mismatch at flat index {index}: got {actual_value}, expected {expected_value}; "
            f"max abs error {max_error}",
        )

    equal = torch.equal(actual_for_compare, expected)
    if equal:
        return True, ""
    mismatch = torch.nonzero((actual_for_compare != expected).reshape(-1), as_tuple=False)
    index = int(mismatch[0].item()) if mismatch.numel() else 0
    actual_value = actual_for_compare.detach().reshape(-1)[index].cpu().item()
    expected_value = expected.detach().reshape(-1)[index].cpu().item()
    return False, f"mismatch at flat index {index}: got {actual_value}, expected {expected_value}"


def tensor_unchanged(actual, original, torch) -> bool:
    if not isinstance(actual, torch.Tensor) or not isinstance(original, torch.Tensor):
        return False
    if tuple(actual.shape) != tuple(original.shape) or actual.dtype != original.dtype:
        return False
    return bool(torch.equal(actual.detach(), original.detach()))


def unchanged_output_hint(key: str, actual, original, torch) -> str:
    if not tensor_unchanged(actual, original, torch):
        return ""
    return (
        f"{key} was unchanged from its initial value. "
        f"Write the computed value into {key} or return a tensor from solve()."
    )


def comparison_failure_hint(challenge: Any, key: str, case: dict[str, Any]) -> str:
    challenge_name = str(getattr(challenge, "name", "")).lower()
    if challenge_name == "logistic regression" and key == "beta":
        samples = case.get("n_samples")
        features = case.get("n_features")
        separable_note = ""
        if isinstance(samples, int) and isinstance(features, int) and samples == features:
            separable_note = (
                " The square test cases can be separable, so unregularized IRLS may keep "
                "moving along a separating direction and produce a different large beta."
            )
        return (
            "This challenge compares beta against its reference Newton/IRLS solver, not only "
            "against the predicted labels. Match the reference details: l2_reg = 1e-6, "
            "W = clamp(p * (1 - p), min=1e-8), gradient = X.T @ (p - y) + l2_reg * beta, "
            "H = X.T @ (X * W[:, None]) + l2_reg * I, and beta -= solve(H, gradient). "
            "If using next_beta, check torch.norm(next_beta - beta) before copying next_beta into beta; "
            "otherwise the loop exits after one iteration."
            f"{separable_note}"
        )
    if challenge_name == "batch normalization" and key == "output":
        return (
            "Batch Normalization normalizes each feature/channel across the batch: "
            "mean = input.mean(dim=0) and variance = input.var(dim=0, unbiased=False). "
            "Using dim=1 computes per-sample normalization, and torch.var defaults to "
            "unbiased=True unless you pass unbiased=False."
        )
    return ""


def run_one_test(
    *,
    name: str,
    raw_case: dict[str, Any],
    challenge: Any,
    solve: Any,
    signature: dict[str, Any],
    output_keys: list[str],
    torch,
    device: str,
    max_test_bytes: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    case = materialize_case(raw_case, torch, device)
    case_summary = summarize_case(case, signature, torch)
    estimated_bytes = case_bytes(case, torch) * 3
    if estimated_bytes > max_test_bytes:
        return {
            "name": name,
            "status": "skipped",
            "durationMs": int((time.perf_counter() - start) * 1000),
            "message": f"Skipped safety-capped case requiring about {estimated_bytes / 1024 / 1024:.1f} MiB",
        }

    reference_case = clone_case(case, torch, device)
    candidate_case = clone_case(case, torch, device)
    initial_output_case = {
        key: clone_value(candidate_case[key], torch, device)
        for key in output_keys
        if isinstance(candidate_case.get(key), torch.Tensor)
    }
    reference_args = args_for_case(signature, reference_case)
    candidate_args = args_for_case(signature, candidate_case)

    try:
        with torch.no_grad():
            challenge.reference_impl(*reference_args)
            synchronize(torch, device)
    except Exception:
        return {
            "name": name,
            "status": "failed",
            "durationMs": int((time.perf_counter() - start) * 1000),
            "message": "Reference implementation failed:\n" + traceback.format_exc(limit=8),
            "case": case_summary,
        }

    try:
        with torch.no_grad():
            returned = solve(*candidate_args)
            synchronize(torch, device)
    except Exception as exc:
        hint = solution_exception_hint(exc, candidate_case, initial_output_case, output_keys, torch)
        message = "Solution raised an exception:\n" + traceback.format_exc(limit=8)
        if hint:
            message = f"{message}\nHint: {hint}"
        return {
            "name": name,
            "status": "failed",
            "durationMs": int((time.perf_counter() - start) * 1000),
            "message": message,
            "outputs": summarize_output_buffers(initial_output_case, output_keys, torch, include_preview=False),
            "case": case_summary,
        }

    overrides = return_overrides(returned, output_keys, torch)
    if not output_keys and isinstance(returned, torch.Tensor):
        output_keys = ["return"]
        reference_case["return"] = returned
        candidate_case["return"] = returned

    output_summaries = []
    for key in output_keys:
        expected = reference_case.get(key)
        actual = overrides.get(key, candidate_case.get(key))
        if not isinstance(expected, torch.Tensor):
            continue
        ok, message = compare_tensors(actual, expected, challenge.atol, challenge.rtol, torch)
        hints = []
        if key not in overrides:
            hint = unchanged_output_hint(key, actual, initial_output_case.get(key), torch)
            if hint:
                hints.append(hint)
        hint = comparison_failure_hint(challenge, key, candidate_case)
        if hint:
            hints.append(hint)
        output_summaries.append(
            {
                "name": key,
                "shape": list(expected.shape),
                "dtype": str(expected.dtype).replace("torch.", ""),
                "preview": tensor_preview(actual, torch) if isinstance(actual, torch.Tensor) else "",
            }
        )
        if not ok:
            if hints:
                message = f"{message}\nHint: {' '.join(hints)}"
            return {
                "name": name,
                "status": "failed",
                "durationMs": int((time.perf_counter() - start) * 1000),
                "message": f"{key}: {message}",
                "outputs": output_summaries,
                "case": case_summary,
            }

    return {
        "name": name,
        "status": "passed",
        "durationMs": int((time.perf_counter() - start) * 1000),
        "message": "Passed",
        "outputs": output_summaries,
    }


def run_request(request: dict[str, Any], captured: io.StringIO, started_at: float) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return result_payload(
            success=False,
            status="error",
            message=f"PyTorch import failed: {exc}",
            output=captured.getvalue(),
            started_at=started_at,
        )

    configure_torch_for_correctness(torch)

    requested_device = str(request.get("device") or "auto")
    try:
        device = pick_device(torch, requested_device)
    except Exception as exc:
        return result_payload(
            success=False,
            status="error",
            message=str(exc),
            output=captured.getvalue(),
            started_at=started_at,
        )

    challenges_root = Path(str(request["challengesRoot"]))
    challenge_dir = Path(str(request["challengeDir"]))
    sys.path.insert(0, str(challenges_root))
    sys.path.insert(0, str(challenge_dir))

    torch.manual_seed(12345)
    if device == "cuda":
        torch.cuda.manual_seed_all(12345)

    try:
        challenge_module = load_module("leetgpu_challenge_under_test", challenge_dir / "challenge.py")
        challenge = challenge_module.Challenge(device=device)
        signature = challenge.get_solve_signature()
        output_keys = output_keys_from_signature(signature)
    except Exception:
        return result_payload(
            success=False,
            status="error",
            message="Failed to load challenge:\n" + traceback.format_exc(limit=8),
            output=captured.getvalue(),
            started_at=started_at,
        )

    solution_path = Path.cwd() / "solution.py"
    solution_path.write_text(str(request.get("code") or ""), encoding="utf-8")
    try:
        solution_globals = {"__name__": "__leetgpu_solution__", "__file__": str(solution_path)}
        exec(compile(solution_path.read_text(encoding="utf-8"), str(solution_path), "exec"), solution_globals)
        solve = solution_globals.get("solve")
        if not callable(solve):
            raise RuntimeError("solution.py must define a callable solve function")
    except Exception:
        return result_payload(
            success=False,
            status="error",
            message="Failed to load solution:\n" + traceback.format_exc(limit=8),
            output=captured.getvalue(),
            started_at=started_at,
            extra={
                "device": device,
                "torchVersion": getattr(torch, "__version__", "unknown"),
                "python": sys.executable,
            },
        )

    tests: list[tuple[str, dict[str, Any]]] = []
    action = str(request.get("action") or "run")
    try:
        for index, case in enumerate(normalize_cases(challenge.generate_example_test()), start=1):
            tests.append((f"Example {index}", case))
        if action == "submit":
            for index, case in enumerate(normalize_cases(challenge.generate_functional_test()), start=1):
                tests.append((f"Functional {index}", case))
    except Exception:
        return result_payload(
            success=False,
            status="error",
            message="Failed to generate tests:\n" + traceback.format_exc(limit=8),
            output=captured.getvalue(),
            started_at=started_at,
            extra={
                "device": device,
                "torchVersion": getattr(torch, "__version__", "unknown"),
                "python": sys.executable,
            },
        )

    max_test_bytes = int(request.get("maxTestBytes") or 1024 * 1024 * 1024)
    test_results = []
    for name, case in tests:
        test_result = run_one_test(
            name=name,
            raw_case=case,
            challenge=challenge,
            solve=solve,
            signature=signature,
            output_keys=list(output_keys),
            torch=torch,
            device=device,
            max_test_bytes=max_test_bytes,
        )
        test_results.append(test_result)
        if test_result["status"] == "failed":
            break

    failed = any(item["status"] == "failed" for item in test_results)
    passed = any(item["status"] == "passed" for item in test_results)
    status = "failed" if failed else "passed"
    if not failed and not passed:
        status = "skipped"
    message = "All executed tests passed" if status == "passed" else "One or more tests failed"
    return result_payload(
        success=(status == "passed"),
        status=status,
        message=message,
        tests=test_results,
        output=captured.getvalue(),
        started_at=started_at,
        extra={
            "device": device,
            "torchVersion": getattr(torch, "__version__", "unknown"),
            "python": sys.executable,
        },
    )


def main() -> int:
    started_at = time.perf_counter()
    apply_resource_limits()
    if len(sys.argv) != 2:
        print(
            json.dumps(
                result_payload(
                    success=False,
                    status="error",
                    message="Usage: judge_worker.py /path/to/request.json",
                    started_at=started_at,
                )
            )
        )
        return 1

    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        payload = run_request(request, captured, started_at)
    print(json.dumps(payload, ensure_ascii=True))
    return 0 if payload.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
