# LeetGPU Local

LeetGPU Local is a self-hosted practice site for GPU programming challenges. It provides a local web UI and judge runtime for solving challenges from the upstream [LeetGPU challenge repository](https://github.com/AlphaGPU/leetgpu-challenges).

Official LeetGPU website: https://leetgpu.com/

## How to Run

Clone with submodules so the challenge dataset is fetched automatically:

```bash
git clone --recurse-submodules <repo-url>
cd leetgpu-local
python3 app.py
```

If you already used a standard `git clone`, initialize the submodule before starting the app:

```bash
git submodule update --init --recursive
python3 app.py
```

Open the printed localhost URL in your browser.

For setup options, storage configuration, and SSH tunneling, see [doc/usage.md](doc/usage.md).

## Challenge Dataset

`leetgpu-challenges/` is tracked as a Git submodule that points to the upstream LeetGPU challenge repository.

## User History

Solved status, attempts, saved drafts, and submissions are user-specific records. They are stored outside the Git repo by default under `~/.local/share/leetgpu-local/`. If you choose to use a project-local `data/` directory, it is ignored by Git.

## Runtime Details

See [docs/runtime.md](docs/runtime.md) for how PyTorch submissions are executed and judged.

## Acknowledgements

The challenge definitions and test data come from the upstream [LeetGPU challenge repository](https://github.com/AlphaGPU/leetgpu-challenges). This project focuses on the local-host web experience, user progress tracking, and runtime judge integration around that dataset.
