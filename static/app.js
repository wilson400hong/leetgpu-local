const app = document.getElementById("app");

const state = {
  challenges: [],
  counts: {},
  runtime: null,
  current: null,
  code: "",
  lastResult: null,
  homeTab: "challenges",
  detailTab: "problem",
  difficulty: "all",
  statusFilter: "all",
  search: "",
  running: false,
  globalSubmissions: [],
  saveTimer: null,
  completion: null,
  bracketHoverIndex: null,
  device: "auto",
  consoleHeight: Math.min(
    520,
    Math.max(120, Number(localStorage.getItem("leetgpu.consoleHeight") || 190) || 190),
  ),
  problemPaneWidthPct: boundedNumber(
    localStorage.getItem("leetgpu.problemPaneWidthPct"),
    50,
    24,
    76,
  ),
};

function boundedNumber(value, fallback, min, max) {
  const number = Number(value);
  const safe = Number.isFinite(number) ? number : fallback;
  return Math.min(max, Math.max(min, safe));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const contentType = response.headers.get("Content-Type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : { error: await response.text() };
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function truncate(value, limit) {
  const text = String(value || "");
  if (text.length <= limit) return text;
  return `${text.slice(0, limit - 3)}...`;
}

const PYTHON_KEYWORDS = new Set([
  "and",
  "as",
  "assert",
  "async",
  "await",
  "break",
  "class",
  "continue",
  "def",
  "del",
  "elif",
  "else",
  "except",
  "finally",
  "for",
  "from",
  "global",
  "if",
  "import",
  "in",
  "is",
  "lambda",
  "nonlocal",
  "not",
  "or",
  "pass",
  "raise",
  "return",
  "try",
  "while",
  "with",
  "yield",
]);

const PYTHON_CONSTANTS = new Set(["False", "None", "True"]);
const PYTHON_BUILTINS = new Set([
  "abs",
  "bool",
  "dict",
  "enumerate",
  "float",
  "int",
  "len",
  "list",
  "max",
  "min",
  "range",
  "round",
  "set",
  "str",
  "sum",
  "tuple",
]);
const PYTHON_MODULES = new Set(["ctypes", "math", "nn", "np", "numpy", "torch"]);
const BRACKET_PAIRS = { "(": ")", "[": "]", "{": "}" };
const BRACKET_OPENERS = new Set(Object.keys(BRACKET_PAIRS));
const BRACKET_CLOSERS = new Set(Object.values(BRACKET_PAIRS));
const BRACKET_CHARS = new Set([...BRACKET_OPENERS, ...BRACKET_CLOSERS]);
const BRACKET_COLOR_COUNT = 6;
const MAX_COMPLETION_ITEMS = 9;
const TORCH_COMPLETIONS = [
  ["abs", "absolute value"],
  ["acos", "inverse cosine"],
  ["add", "elementwise addition"],
  ["arange", "range tensor"],
  ["argmax", "index of maximum"],
  ["argmin", "index of minimum"],
  ["argsort", "sorted indices"],
  ["bmm", "batched matrix multiply"],
  ["cat", "concatenate tensors"],
  ["clamp", "clamp values"],
  ["cos", "cosine"],
  ["cumprod", "cumulative product"],
  ["cumsum", "cumulative sum"],
  ["div", "elementwise division"],
  ["dot", "vector dot product"],
  ["empty", "uninitialized tensor"],
  ["empty_like", "uninitialized tensor like input"],
  ["einsum", "Einstein summation"],
  ["eq", "elementwise equality"],
  ["exp", "exponential"],
  ["eye", "identity matrix"],
  ["flatten", "flatten tensor"],
  ["floor", "floor values"],
  ["full", "filled tensor"],
  ["full_like", "filled tensor like input"],
  ["gather", "gather values"],
  ["isclose", "elementwise closeness"],
  ["log", "natural logarithm"],
  ["log_softmax", "log softmax"],
  ["logsumexp", "stable log-sum-exp"],
  ["matmul", "matrix product"],
  ["max", "maximum"],
  ["mean", "mean value"],
  ["min", "minimum"],
  ["mm", "matrix multiply"],
  ["mul", "elementwise multiplication"],
  ["no_grad", "disable autograd"],
  ["nonzero", "indices of nonzero values"],
  ["ones", "ones tensor"],
  ["ones_like", "ones tensor like input"],
  ["prod", "product"],
  ["rand", "uniform random tensor"],
  ["randint", "integer random tensor"],
  ["randn", "normal random tensor"],
  ["reshape", "reshape tensor"],
  ["roll", "roll tensor"],
  ["round", "round values"],
  ["rsqrt", "reciprocal square root"],
  ["scatter", "scatter values"],
  ["sigmoid", "sigmoid"],
  ["sin", "sine"],
  ["softmax", "softmax"],
  ["sort", "sort tensor"],
  ["sqrt", "square root"],
  ["stack", "stack tensors"],
  ["sum", "sum values"],
  ["take_along_dim", "take values along dimension"],
  ["tan", "tangent"],
  ["tensor", "construct tensor"],
  ["topk", "top k values"],
  ["transpose", "swap two dimensions"],
  ["where", "conditional select"],
  ["zeros", "zeros tensor"],
  ["zeros_like", "zeros tensor like input"],
].map(([name, detail]) => ({ name, detail }));

function tokenSpan(className, value) {
  return `<span class="${className}">${escapeHtml(value)}</span>`;
}

function bracketSpan(value, info) {
  const classes = [
    "tok-bracket",
    `bracket-depth-${Math.abs(info.depth) % BRACKET_COLOR_COUNT}`,
  ];
  if (!info.matched) classes.push("bracket-unmatched");
  if (info.active) classes.push("bracket-active");
  return tokenSpan(classes.join(" "), value);
}

function highlightPython(code, bracketInfo = new Map()) {
  let output = "";
  let index = 0;
  let expectDeclaration = false;

  while (index < code.length) {
    const char = code[index];

    if (char === "#") {
      const end = code.indexOf("\n", index);
      const next = end === -1 ? code.length : end;
      output += tokenSpan("tok-comment", code.slice(index, next));
      index = next;
      continue;
    }

    if (char === '"' || char === "'") {
      const parsed = readPythonString(code, index);
      output += tokenSpan("tok-string", parsed.value);
      index = parsed.end;
      continue;
    }

    if (isDigit(char) || (char === "." && isDigit(code[index + 1]))) {
      const match = code.slice(index).match(/^(0[xX][0-9a-fA-F_]+|\d[\d_]*(\.\d[\d_]*)?([eE][+-]?\d[\d_]*)?|\.\d[\d_]*([eE][+-]?\d[\d_]*)?)/);
      if (match) {
        output += tokenSpan("tok-number", match[0]);
        index += match[0].length;
        continue;
      }
    }

    if (isIdentifierStart(char)) {
      let end = index + 1;
      while (end < code.length && isIdentifierPart(code[end])) end += 1;
      const word = code.slice(index, end);
      const next = nextNonSpace(code, end);
      const previous = previousNonSpace(code, index);

      if (expectDeclaration) {
        output += tokenSpan("tok-function", word);
        expectDeclaration = false;
      } else if (PYTHON_KEYWORDS.has(word)) {
        output += tokenSpan("tok-keyword", word);
        expectDeclaration = word === "def" || word === "class";
      } else if (PYTHON_CONSTANTS.has(word)) {
        output += tokenSpan("tok-constant", word);
      } else if (PYTHON_MODULES.has(word)) {
        output += tokenSpan("tok-module", word);
      } else if (PYTHON_BUILTINS.has(word)) {
        output += tokenSpan("tok-builtin", word);
      } else if (previous === "." && /^[A-Z]/.test(word)) {
        output += tokenSpan("tok-type", word);
      } else if (next === "(") {
        output += tokenSpan("tok-call", word);
      } else {
        output += escapeHtml(word);
      }
      index = end;
      continue;
    }

    const bracket = bracketInfo.get(index);
    output += bracket ? bracketSpan(char, bracket) : escapeHtml(char);
    index += 1;
  }

  return output || " ";
}

function analyzePythonBrackets(code, activeIndex) {
  const stack = [];
  const pairs = [];
  const unmatched = [];
  let index = 0;

  while (index < code.length) {
    const char = code[index];

    if (char === "#") {
      const end = code.indexOf("\n", index);
      index = end === -1 ? code.length : end;
      continue;
    }

    if (char === '"' || char === "'") {
      index = readPythonString(code, index).end;
      continue;
    }

    if (BRACKET_OPENERS.has(char)) {
      stack.push({ char, index, depth: stack.length });
      index += 1;
      continue;
    }

    if (BRACKET_CLOSERS.has(char)) {
      const opener = stack[stack.length - 1];
      if (opener && BRACKET_PAIRS[opener.char] === char) {
        stack.pop();
        pairs.push({ open: opener.index, close: index, depth: opener.depth });
      } else {
        unmatched.push({ index, depth: stack.length });
      }
      index += 1;
      continue;
    }

    index += 1;
  }

  stack.forEach((opener) => unmatched.push({ index: opener.index, depth: opener.depth }));

  const activePair =
    activeIndex == null
      ? null
      : pairs.find((pair) => pair.open === activeIndex || pair.close === activeIndex);
  const info = new Map();

  pairs.forEach((pair) => {
    const active = activePair === pair;
    const value = { depth: pair.depth, matched: true, active };
    info.set(pair.open, value);
    info.set(pair.close, value);
  });
  unmatched.forEach((item) => {
    info.set(item.index, { depth: item.depth, matched: false, active: item.index === activeIndex });
  });

  return info;
}

function readPythonString(code, start) {
  const quote = code[start];
  const triple = code.slice(start, start + 3) === quote.repeat(3);
  let index = start + (triple ? 3 : 1);

  while (index < code.length) {
    if (code[index] === "\\") {
      index += 2;
      continue;
    }
    if (triple && code.slice(index, index + 3) === quote.repeat(3)) {
      return { value: code.slice(start, index + 3), end: index + 3 };
    }
    if (!triple && code[index] === quote) {
      return { value: code.slice(start, index + 1), end: index + 1 };
    }
    index += 1;
  }

  return { value: code.slice(start), end: code.length };
}

function isDigit(char) {
  return Boolean(char && /[0-9]/.test(char));
}

function isIdentifierStart(char) {
  return Boolean(char && /[A-Za-z_]/.test(char));
}

function isIdentifierPart(char) {
  return Boolean(char && /[A-Za-z0-9_]/.test(char));
}

function nextNonSpace(code, start) {
  let index = start;
  while (index < code.length && /[ \t]/.test(code[index])) index += 1;
  return code[index] || "";
}

function previousNonSpace(code, start) {
  let index = start - 1;
  while (index >= 0 && /[ \t]/.test(code[index])) index -= 1;
  return code[index] || "";
}

function consoleLine(className, text) {
  return `<span class="${className}">${escapeHtml(text)}</span>`;
}

function consoleStatusClass(status) {
  if (status === "passed" || status === "success" || status === "solved") return "console-success";
  if (status === "failed" || status === "error" || status === "timeout") return "console-failed";
  if (status === "skipped") return "console-warning";
  return "console-muted";
}

function route() {
  closeCompletionMenu();
  closeNextChallengePrompt();
  const hash = window.location.hash || "#/";
  const match = hash.match(/^#\/challenge\/(.+)$/);
  if (match) {
    loadChallenge(decodeURIComponent(match[1]));
    return;
  }
  state.current = null;
  state.lastResult = null;
  renderHome();
}

async function loadRuntime() {
  try {
    state.runtime = await api("/api/runtime");
  } catch (error) {
    state.runtime = { ok: false, error: error.message };
  }
}

async function loadChallenges() {
  const payload = await api("/api/challenges");
  state.challenges = payload.challenges || [];
  state.counts = payload.counts || {};
}

async function loadChallenge(id) {
  renderLoading("Loading challenge...");
  try {
    state.current = await api(`/api/challenge?id=${encodeURIComponent(id)}`);
    state.code = state.current.code || state.current.starter || "";
    state.lastResult = null;
    state.bracketHoverIndex = null;
    renderChallenge();
  } catch (error) {
    renderError(error.message);
  }
}

async function loadGlobalSubmissions() {
  try {
    const payload = await api("/api/submissions");
    state.globalSubmissions = payload.submissions || [];
  } catch {
    state.globalSubmissions = [];
  }
}

function runtimeLabel() {
  if (!state.runtime) return "Runtime checking";
  if (!state.runtime.ok) return "PyTorch unavailable";
  const device = state.runtime.device || "CPU";
  const version = state.runtime.torchVersion || "unknown";
  return `${device} | PyTorch ${version}`;
}

function renderShell(content) {
  app.innerHTML = `
    <header class="topbar">
      <button class="brand-button" data-home aria-label="Open challenges">LeetGPU Local</button>
      <div class="topbar-actions">
        <div class="runtime-chip" title="${escapeHtml(state.runtime?.python || "")}">
          <span class="chip-dot ${state.runtime?.ok ? "ok" : "bad"}"></span>
          <span>${escapeHtml(runtimeLabel())}</span>
        </div>
        ${state.current ? editorActionButtons() : ""}
        <button class="ghost-button" data-refresh>Refresh</button>
      </div>
    </header>
    ${content}
  `;
  bindShell();
}

function editorActionButtons() {
  return `
    <select class="select" data-device aria-label="Device">
      <option value="auto" ${state.device === "auto" ? "selected" : ""}>Auto device</option>
      <option value="cuda" ${state.device === "cuda" ? "selected" : ""}>CUDA</option>
      <option value="cpu" ${state.device === "cpu" ? "selected" : ""}>CPU</option>
    </select>
    <select class="select" aria-label="Language" disabled>
      <option>PyTorch</option>
    </select>
    <button class="primary-button blue" data-run ${state.running ? "disabled" : ""}>Run</button>
    <button class="primary-button green" data-submit ${state.running ? "disabled" : ""}>Submit</button>
  `;
}

function renderLoading(message) {
  renderShell(`
    <main class="main-content">
      <div class="empty-state">${escapeHtml(message)}</div>
    </main>
  `);
}

function renderError(message) {
  renderShell(`
    <main class="main-content">
      <div class="error-state">${escapeHtml(message)}</div>
    </main>
  `);
}

function renderHome() {
  const filtered = filteredChallenges();
  const totals = totalCounts();
  renderShell(`
    <main class="main-content">
      <section class="status-strip">
        <div class="status-icon">GPU</div>
        <div>
          <div class="status-title">Local PyTorch Judge</div>
          <div class="status-subtitle">${escapeHtml(runtimeLabel())}</div>
        </div>
        <div class="status-metrics">
          <span>${totals.solved} solved</span>
          <span>${totals.tried} tried</span>
          <span>${totals.total} challenges</span>
        </div>
      </section>

      <nav class="tabs">
        <button class="${state.homeTab === "challenges" ? "active" : ""}" data-home-tab="challenges">Challenges</button>
        <button class="${state.homeTab === "submissions" ? "active" : ""}" data-home-tab="submissions">Submissions</button>
      </nav>

      ${state.homeTab === "submissions" ? renderGlobalSubmissions() : renderChallengeList(filtered)}
    </main>
  `);
  bindHome();
}

function renderChallengeList(filtered) {
  return `
    <section class="list-header">
      <div>
        <h1>Challenges</h1>
        <p>${filtered.length} shown</p>
      </div>
      <div class="filters">
        <input class="search" data-search type="search" value="${escapeHtml(state.search)}" placeholder="Search challenges..." />
        <div class="segmented" aria-label="Difficulty filter">
          ${filterButton("all", "All", state.difficulty)}
          ${filterButton("easy", "Easy", state.difficulty)}
          ${filterButton("medium", "Medium", state.difficulty)}
          ${filterButton("hard", "Hard", state.difficulty)}
        </div>
        <select class="select" data-status-select aria-label="Status filter">
          <option value="all" ${state.statusFilter === "all" ? "selected" : ""}>All Challenges</option>
          <option value="solved" ${state.statusFilter === "solved" ? "selected" : ""}>Solved</option>
          <option value="tried" ${state.statusFilter === "tried" ? "selected" : ""}>Tried</option>
          <option value="untried" ${state.statusFilter === "untried" ? "selected" : ""}>Untried</option>
        </select>
      </div>
    </section>
    <section class="challenge-grid" id="challengeGrid">
      ${challengeCards(filtered)}
    </section>
  `;
}

function renderGlobalSubmissions() {
  return `
    <section class="submissions-panel wide">
      <div class="panel-title">Recent Submissions</div>
      ${submissionRows(state.globalSubmissions, true)}
    </section>
  `;
}

function filterButton(value, label, activeValue) {
  return `<button class="${activeValue === value ? "active" : ""}" data-difficulty="${value}">${label}</button>`;
}

function totalCounts() {
  return state.challenges.reduce(
    (acc, item) => {
      acc.total += 1;
      if (item.status === "solved") acc.solved += 1;
      if (item.status === "tried") acc.tried += 1;
      return acc;
    },
    { total: 0, solved: 0, tried: 0 },
  );
}

function filteredChallenges() {
  const query = state.search.trim().toLowerCase();
  return state.challenges.filter((item) => {
    if (state.difficulty !== "all" && item.difficulty !== state.difficulty) return false;
    if (state.statusFilter !== "all" && item.status !== state.statusFilter) return false;
    if (!query) return true;
    return (
      item.title.toLowerCase().includes(query) ||
      item.description.toLowerCase().includes(query) ||
      item.slug.toLowerCase().includes(query)
    );
  });
}

function challengeCards(items) {
  if (!items.length) {
    return `<div class="empty-state grid-empty">No challenges match the current filters.</div>`;
  }
  return items
    .map(
      (item) => {
        const href = challengeHref(item.id);
        return `
        <a class="challenge-card" data-challenge-id="${escapeHtml(item.id)}" href="${escapeHtml(href)}">
          <div class="card-meta">
            <span class="difficulty ${item.difficulty}">${escapeHtml(capitalize(item.difficulty))}</span>
            <span class="status-pill ${item.status}">${statusLabel(item.status)}</span>
          </div>
          <h2>${escapeHtml(item.title)}</h2>
          <p>${escapeHtml(truncate(item.description, 145))}</p>
        </a>
      `;
      },
    )
    .join("");
}

function challengeHref(id) {
  return `#/challenge/${encodeURIComponent(id)}`;
}

function nextUnsolvedChallenge(currentId) {
  if (!state.challenges.length) return null;
  const currentIndex = state.challenges.findIndex((item) => item.id === currentId);
  const startIndex = currentIndex === -1 ? 0 : currentIndex + 1;

  for (let offset = 0; offset < state.challenges.length; offset += 1) {
    const item = state.challenges[(startIndex + offset) % state.challenges.length];
    if (item.id !== currentId && item.status !== "solved") return item;
  }

  return null;
}

function capitalize(value) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

function statusLabel(status) {
  if (status === "solved") return "Solved";
  if (status === "tried") return "Tried";
  return "Open";
}

function renderChallenge() {
  const item = state.current;
  if (!item) return renderHome();
  renderShell(`
    <main class="workspace" style="--problem-pane-width: ${state.problemPaneWidthPct}%">
      <aside class="left-pane">
        <div class="challenge-title-row">
          <button class="back-button" data-home aria-label="Back to challenges">&lt;</button>
          <div>
            <div class="eyebrow">${escapeHtml(capitalize(item.difficulty))} / #${item.number}</div>
            <h1>${escapeHtml(item.title)}</h1>
          </div>
          <span class="status-pill ${item.status}">${statusLabel(item.status)}</span>
        </div>
        <nav class="tabs compact">
          <button class="${state.detailTab === "problem" ? "active" : ""}" data-detail-tab="problem">Problem</button>
          <button class="${state.detailTab === "submissions" ? "active" : ""}" data-detail-tab="submissions">Submissions</button>
        </nav>
        <div class="left-scroll">
          ${state.detailTab === "submissions" ? renderDetailSubmissions(item) : renderProblem(item)}
        </div>
      </aside>
      <div class="pane-resizer" data-pane-resizer role="separator" aria-orientation="vertical" aria-label="Resize problem and IDE panes" title="Drag to resize panes">
        <span></span>
      </div>
      <section class="right-pane" style="--console-height: ${state.consoleHeight}px">
        <div class="editor-bar">
          <div class="file-name">solution.py <span class="public-chip">Local</span></div>
          <div class="save-state" id="saveState">Saved</div>
        </div>
        <div class="editor-wrap">
          <pre class="line-numbers" id="lineNumbers"></pre>
          <div class="editor-stage">
            <pre class="code-highlight" id="codeHighlight" aria-hidden="true"></pre>
            <textarea id="codeEditor" class="code-editor" spellcheck="false" wrap="off">${escapeHtml(state.code)}</textarea>
            <div id="completionMenu" class="completion-menu" hidden></div>
          </div>
        </div>
        <div class="console-resizer" data-console-resizer role="separator" aria-label="Resize console" title="Drag to resize console">
          <span></span>
        </div>
        <section class="console">
          <div class="console-header">
            <span>Console Output</span>
            ${state.lastResult ? `<span class="${consoleStatusClass(state.lastResult.status)}">${escapeHtml(state.lastResult.status || "")}</span>` : ""}
          </div>
          <pre id="consoleOutput" class="console-output">${consoleHtml()}</pre>
        </section>
      </section>
    </main>
  `);
  bindChallenge();
  typesetMath();
}

function renderProblem(item) {
  return `
    <article class="problem-content">
      ${item.html}
    </article>
  `;
}

function renderDetailSubmissions(item) {
  return `
    <section class="submissions-panel">
      ${submissionRows(item.submissions || [], false)}
    </section>
  `;
}

function submissionRows(rows, showChallenge) {
  if (!rows.length) {
    return `<div class="empty-state">No submissions available.</div>`;
  }
  return `
    <div class="submission-list">
      ${rows
        .map((row) => {
          const challenge = state.challenges.find((item) => item.id === row.challenge_id);
          return `
            <div class="submission-row">
              <span class="submission-status ${row.status}">${escapeHtml(row.status)}</span>
              <span>${escapeHtml(row.action)}</span>
              ${showChallenge ? `<span>${escapeHtml(challenge?.title || row.challenge_id)}</span>` : ""}
              <span>${row.passed} passed</span>
              <span>${row.failed} failed</span>
              <span>${row.skipped} skipped</span>
              <span>${row.duration_ms} ms</span>
              <time>${formatTime(row.created_at)}</time>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function formatTime(value) {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function consoleHtml() {
  if (state.running) return `${consoleLine("console-muted", "Spinning up runner...")}\n`;
  if (!state.lastResult) return `${consoleLine("console-muted", "Ready.")}\n`;
  const result = state.lastResult;
  const lines = [];
  const status = result.status || "result";
  lines.push(
    consoleLine(
      consoleStatusClass(status),
      `${status.toUpperCase()}: ${result.message || ""}`,
    ),
  );
  if (result.device || result.torchVersion) {
    lines.push(
      consoleLine(
        "console-muted",
        `Runtime: ${result.device || "unknown"} | PyTorch ${result.torchVersion || "unknown"}`,
      ),
    );
  }
  const summary = result.summary || {};
  lines.push(
    `${consoleLine("console-muted", "Tests:")} ${consoleLine("console-success", `${summary.passed || 0} passed`)}, ${consoleLine("console-failed", `${summary.failed || 0} failed`)}, ${consoleLine("console-warning", `${summary.skipped || 0} skipped`)} ${consoleLine("console-muted", `in ${result.durationMs || 0} ms`)}`,
  );
  (result.tests || []).forEach((test) => {
    lines.push("");
    lines.push(
      consoleLine(
        consoleStatusClass(test.status),
        `[${test.status}] ${test.name} (${test.durationMs || 0} ms)`,
      ),
    );
    if (test.message) lines.push(consoleLine(consoleStatusClass(test.status), test.message.trim()));
    if (test.status === "failed" && test.case?.length) {
      lines.push(consoleLine("console-muted", "Failed test case:"));
      test.case.forEach((entry) => {
        lines.push(`  ${caseEntryHtml(entry)}`);
      });
    }
    (test.outputs || []).forEach((output) => {
      lines.push(
        `  ${consoleLine("console-muted", `${output.name}:`)} ${escapeHtml(output.dtype)} ${escapeHtml(JSON.stringify(output.shape))} ${escapeHtml(output.preview || "")}`,
      );
    });
  });
  if (result.output) {
    lines.push("");
    lines.push(consoleLine("console-muted", "Captured output:"));
    lines.push(escapeHtml(result.output));
  }
  if (result.workerStderr) {
    lines.push("");
    lines.push(consoleLine("console-muted", "Worker stderr:"));
    lines.push(consoleLine("console-failed", result.workerStderr));
  }
  return `${lines.join("\n")}\n`;
}

function caseEntryHtml(entry) {
  const name = consoleLine("console-case-name", entry.name || "value");
  const direction = entry.direction ? ` ${consoleLine("console-muted", `(${entry.direction})`)}` : "";
  if (entry.kind === "tensor") {
    return `${name}${direction}: ${escapeHtml(entry.dtype)} ${escapeHtml(JSON.stringify(entry.shape || []))}, preview ${escapeHtml(entry.preview || "[]")}`;
  }
  if (entry.kind === "module") {
    return `${name}${direction}: ${escapeHtml(entry.type || "Module")} with ${escapeHtml(entry.parameters || 0)} parameters`;
  }
  return `${name}${direction}: ${escapeHtml(entry.value || "")}`;
}

function bindShell() {
  document.querySelectorAll("[data-home]").forEach((button) => {
    button.addEventListener("click", () => {
      window.location.hash = "#/";
    });
  });
  document.querySelectorAll("[data-refresh]").forEach((button) => {
    button.addEventListener("click", async () => {
      await refreshData();
      route();
    });
  });
  document.querySelectorAll("[data-run]").forEach((button) => {
    button.addEventListener("click", () => judge("run"));
  });
  document.querySelectorAll("[data-submit]").forEach((button) => {
    button.addEventListener("click", () => judge("submit"));
  });
  const device = document.querySelector("[data-device]");
  if (device) {
    device.addEventListener("change", () => {
      state.device = device.value;
    });
  }
}

function bindHome() {
  document.querySelectorAll("[data-home-tab]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.homeTab = button.dataset.homeTab;
      if (state.homeTab === "submissions") await loadGlobalSubmissions();
      renderHome();
    });
  });
  document.querySelectorAll("[data-difficulty]").forEach((button) => {
    button.addEventListener("click", () => {
      state.difficulty = button.dataset.difficulty;
      renderHome();
    });
  });
  const status = document.querySelector("[data-status-select]");
  if (status) {
    status.addEventListener("change", () => {
      state.statusFilter = status.value;
      renderHome();
    });
  }
  const search = document.querySelector("[data-search]");
  if (search) {
    search.addEventListener("input", () => {
      state.search = search.value;
      const grid = document.getElementById("challengeGrid");
      if (grid) grid.innerHTML = challengeCards(filteredChallenges());
      bindChallengeCards();
    });
  }
  bindChallengeCards();
}

function bindChallengeCards() {
  document.querySelectorAll("[data-challenge-id]").forEach((card) => {
    card.addEventListener("keydown", (event) => {
      if (event.key === " ") {
        event.preventDefault();
        window.location.href = card.href;
      }
    });
  });
}

function bindChallenge() {
  document.querySelectorAll("[data-detail-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.detailTab = button.dataset.detailTab;
      renderChallenge();
    });
  });
  const editor = document.getElementById("codeEditor");
  const lines = document.getElementById("lineNumbers");
  const highlight = document.getElementById("codeHighlight");
  const completionMenu = document.getElementById("completionMenu");
  bindPaneResizer();
  bindConsoleResizer();
  if (!editor || !lines || !highlight || !completionMenu) return;
  editor.value = state.code;
  updateEditorDecorations(editor, lines, highlight);
  editor.addEventListener("input", () => {
    state.code = editor.value;
    updateEditorDecorations(editor, lines, highlight);
    updateCompletionMenu(editor, completionMenu);
    scheduleSave();
  });
  editor.addEventListener("scroll", () => {
    syncEditorScroll(editor, lines, highlight);
    positionCompletionMenu(editor, completionMenu);
  });
  editor.addEventListener("click", () => {
    updateEditorDecorations(editor, lines, highlight);
    updateCompletionMenu(editor, completionMenu);
  });
  editor.addEventListener("keyup", (event) => {
    updateEditorDecorations(editor, lines, highlight);
    if (["ArrowUp", "ArrowDown", "Enter", "Escape", "Tab"].includes(event.key)) return;
    updateCompletionMenu(editor, completionMenu);
  });
  editor.addEventListener("select", () => {
    updateEditorDecorations(editor, lines, highlight);
  });
  editor.addEventListener("mousemove", (event) => {
    const hoverIndex = bracketIndexFromEditorPoint(editor, event.clientX, event.clientY);
    if (hoverIndex === state.bracketHoverIndex) return;
    state.bracketHoverIndex = hoverIndex;
    updateEditorDecorations(editor, lines, highlight);
  });
  editor.addEventListener("mouseleave", () => {
    if (state.bracketHoverIndex == null) return;
    state.bracketHoverIndex = null;
    updateEditorDecorations(editor, lines, highlight);
  });
  editor.addEventListener("blur", () => {
    state.bracketHoverIndex = null;
    updateEditorDecorations(editor, lines, highlight);
    window.setTimeout(() => closeCompletionMenu(completionMenu), 120);
  });
  editor.addEventListener("keydown", (event) => {
    if (isCompletionShortcut(event)) {
      event.preventDefault();
      updateCompletionMenu(editor, completionMenu);
      return;
    }
    if (event.key === ".") {
      window.setTimeout(() => updateCompletionMenu(editor, completionMenu), 0);
    }
    if (handleCompletionKeydown(event, editor, lines, highlight, completionMenu)) {
      return;
    }
    if (isCommentShortcut(event)) {
      event.preventDefault();
      closeCompletionMenu(completionMenu);
      toggleLineComments(editor, lines, highlight);
      return;
    }
    if (isLineCutShortcut(event) && editor.selectionStart === editor.selectionEnd) {
      event.preventDefault();
      closeCompletionMenu(completionMenu);
      removeCurrentLine(editor, lines, highlight);
      return;
    }
    if (event.key === "Tab") {
      event.preventDefault();
      const start = editor.selectionStart;
      const end = editor.selectionEnd;
      const spaces = "    ";
      editor.setRangeText(spaces, start, end, "end");
      closeCompletionMenu(completionMenu);
      applyEditorChange(editor, lines, highlight);
    }
  });
  bindCompletionMouse(editor, lines, highlight, completionMenu);
}

function isEditorModifier(event) {
  return event.metaKey || event.ctrlKey;
}

function isCommentShortcut(event) {
  return (
    isEditorModifier(event) &&
    !event.altKey &&
    !event.shiftKey &&
    (event.key === "/" || event.code === "Slash")
  );
}

function isLineCutShortcut(event) {
  return isEditorModifier(event) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === "x";
}

function isCompletionShortcut(event) {
  return isEditorModifier(event) && !event.altKey && !event.shiftKey && event.code === "Space";
}

function currentTorchCompletionContext(editor) {
  if (editor.selectionStart !== editor.selectionEnd) return null;

  const cursor = editor.selectionStart;
  const before = editor.value.slice(0, cursor);
  const trigger = before.match(/(?:^|[^A-Za-z0-9_.])torch\.([A-Za-z_]*)$/);
  if (!trigger) return null;

  const typedPrefix = trigger[1] || "";
  const triggerText = trigger[0];
  const torchIndex = triggerText.lastIndexOf("torch.");
  const rangeStart = cursor - triggerText.length + torchIndex + "torch.".length;
  const suffix = editor.value.slice(cursor).match(/^[A-Za-z0-9_]*/)?.[0] || "";
  return {
    prefix: typedPrefix,
    rangeStart,
    rangeEnd: cursor + suffix.length,
  };
}

function torchCompletionMatches(prefix) {
  const normalized = prefix.toLowerCase();
  const startsWith = TORCH_COMPLETIONS.filter((item) =>
    item.name.toLowerCase().startsWith(normalized),
  );
  const contains =
    normalized.length > 0
      ? TORCH_COMPLETIONS.filter(
          (item) =>
            !item.name.toLowerCase().startsWith(normalized) &&
            item.name.toLowerCase().includes(normalized),
        )
      : [];
  return [...startsWith, ...contains].slice(0, MAX_COMPLETION_ITEMS);
}

function updateCompletionMenu(editor, menu, selectedIndex = 0) {
  const context = currentTorchCompletionContext(editor);
  if (!context) {
    closeCompletionMenu(menu);
    return;
  }

  const items = torchCompletionMatches(context.prefix);
  if (!items.length) {
    closeCompletionMenu(menu);
    return;
  }

  state.completion = {
    ...context,
    items,
    selectedIndex: boundedNumber(selectedIndex, 0, 0, items.length - 1),
  };
  renderCompletionMenu(menu);
  positionCompletionMenu(editor, menu);
}

function renderCompletionMenu(menu) {
  const completion = state.completion;
  if (!completion?.items?.length) {
    closeCompletionMenu(menu);
    return;
  }

  menu.hidden = false;
  menu.innerHTML = completion.items
    .map((item, index) => {
      const active = index === completion.selectedIndex ? "active" : "";
      return `
        <button class="completion-item ${active}" data-completion-index="${index}" type="button">
          <span class="completion-name">torch.${completionNameHtml(item.name, completion.prefix)}</span>
          <span class="completion-detail">${escapeHtml(item.detail)}</span>
        </button>
      `;
    })
    .join("");
  menu.querySelector(".completion-item.active")?.scrollIntoView({ block: "nearest" });
}

function completionNameHtml(name, prefix) {
  if (!prefix) return escapeHtml(name);
  const lowerName = name.toLowerCase();
  const lowerPrefix = prefix.toLowerCase();
  const index = lowerName.indexOf(lowerPrefix);
  if (index === -1) return escapeHtml(name);
  return `${escapeHtml(name.slice(0, index))}<span>${escapeHtml(
    name.slice(index, index + prefix.length),
  )}</span>${escapeHtml(name.slice(index + prefix.length))}`;
}

function closeCompletionMenu(menu = document.getElementById("completionMenu")) {
  state.completion = null;
  if (!menu) return;
  menu.hidden = true;
  menu.innerHTML = "";
}

function handleCompletionKeydown(event, editor, lines, highlight, menu) {
  const completion = state.completion;
  if (!completion || menu.hidden) return false;
  if (event.metaKey || event.ctrlKey || event.altKey) return false;

  if (event.key === "Escape") {
    event.preventDefault();
    closeCompletionMenu(menu);
    return true;
  }
  if (event.key === "ArrowDown" || event.key === "ArrowUp") {
    event.preventDefault();
    const direction = event.key === "ArrowDown" ? 1 : -1;
    const nextIndex =
      (completion.selectedIndex + direction + completion.items.length) % completion.items.length;
    state.completion = { ...completion, selectedIndex: nextIndex };
    renderCompletionMenu(menu);
    positionCompletionMenu(editor, menu);
    return true;
  }
  if (event.key === "Enter" || event.key === "Tab") {
    event.preventDefault();
    acceptCompletion(editor, lines, highlight, menu);
    return true;
  }

  return false;
}

function bindCompletionMouse(editor, lines, highlight, menu) {
  menu.addEventListener("mousedown", (event) => {
    const button = event.target.closest("[data-completion-index]");
    if (!button) return;
    event.preventDefault();
    const selectedIndex = Number(button.dataset.completionIndex);
    if (Number.isFinite(selectedIndex) && state.completion) {
      state.completion = { ...state.completion, selectedIndex };
      acceptCompletion(editor, lines, highlight, menu);
    }
  });
}

function acceptCompletion(editor, lines, highlight, menu) {
  const completion = state.completion;
  const item = completion?.items?.[completion.selectedIndex];
  if (!completion || !item) return;

  editor.setRangeText(item.name, completion.rangeStart, completion.rangeEnd, "end");
  applyEditorChange(editor, lines, highlight);
  closeCompletionMenu(menu);
  editor.focus();
}

function positionCompletionMenu(editor, menu) {
  const completion = state.completion;
  if (!completion || menu.hidden) return;

  const style = window.getComputedStyle(editor);
  const lineHeight = Number.parseFloat(style.lineHeight) || 21;
  const paddingLeft = Number.parseFloat(style.paddingLeft) || 0;
  const paddingTop = Number.parseFloat(style.paddingTop) || 0;
  const before = editor.value.slice(0, editor.selectionStart);
  const lineIndex = before.split("\n").length - 1;
  const lineStart = before.lastIndexOf("\n") + 1;
  const visualColumn = editorVisualColumn(before.slice(lineStart), Number.parseInt(style.tabSize, 10) || 4);
  const charWidth = editorCharWidth(editor);
  const stage = editor.parentElement;
  const rawLeft = paddingLeft + visualColumn * charWidth - editor.scrollLeft;
  const rawTop = paddingTop + (lineIndex + 1) * lineHeight - editor.scrollTop + 4;
  const maxLeft = Math.max(8, (stage?.clientWidth || editor.clientWidth) - menu.offsetWidth - 8);
  const belowFits = rawTop + menu.offsetHeight <= (stage?.clientHeight || editor.clientHeight) - 8;
  const top = belowFits ? rawTop : paddingTop + lineIndex * lineHeight - editor.scrollTop - menu.offsetHeight - 4;

  menu.style.left = `${Math.max(8, Math.min(rawLeft, maxLeft))}px`;
  menu.style.top = `${Math.max(8, top)}px`;
}

function editorVisualColumn(text, tabSize) {
  let column = 0;
  for (const char of text) {
    column += char === "\t" ? tabSize - (column % tabSize) : 1;
  }
  return column;
}

function editorCharWidth(editor) {
  const style = window.getComputedStyle(editor);
  const canvas = editorCharWidth.canvas || document.createElement("canvas");
  editorCharWidth.canvas = canvas;
  const context = canvas.getContext("2d");
  context.font = `${style.fontStyle} ${style.fontVariant} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  return context.measureText("M").width || 8;
}

function activeBracketIndex(editor) {
  if (Number.isInteger(state.bracketHoverIndex) && BRACKET_CHARS.has(editor.value[state.bracketHoverIndex])) {
    return state.bracketHoverIndex;
  }

  if (editor.selectionStart !== editor.selectionEnd) return null;
  const cursor = editor.selectionStart;
  if (BRACKET_CHARS.has(editor.value[cursor])) return cursor;
  if (cursor > 0 && BRACKET_CHARS.has(editor.value[cursor - 1])) return cursor - 1;
  return null;
}

function bracketIndexFromEditorPoint(editor, clientX, clientY) {
  const index = editorIndexFromPoint(editor, clientX, clientY);
  if (index == null) return null;
  return BRACKET_CHARS.has(editor.value[index]) ? index : null;
}

function editorIndexFromPoint(editor, clientX, clientY) {
  const rect = editor.getBoundingClientRect();
  const style = window.getComputedStyle(editor);
  const paddingLeft = Number.parseFloat(style.paddingLeft) || 0;
  const paddingTop = Number.parseFloat(style.paddingTop) || 0;
  const lineHeight = Number.parseFloat(style.lineHeight) || 21;
  const charWidth = editorCharWidth(editor);
  const tabSize = Number.parseInt(style.tabSize, 10) || 4;
  const x = clientX - rect.left - paddingLeft + editor.scrollLeft;
  const y = clientY - rect.top - paddingTop + editor.scrollTop;
  if (x < 0 || y < 0) return null;

  const lineIndex = Math.floor(y / lineHeight);
  const targetColumn = Math.floor(x / charWidth);
  const lines = editor.value.split("\n");
  if (lineIndex < 0 || lineIndex >= lines.length) return null;

  let lineStart = 0;
  for (let index = 0; index < lineIndex; index += 1) {
    lineStart += lines[index].length + 1;
  }

  return lineStart + lineIndexFromVisualColumn(lines[lineIndex], targetColumn, tabSize);
}

function lineIndexFromVisualColumn(line, targetColumn, tabSize) {
  let column = 0;
  for (let index = 0; index < line.length; index += 1) {
    const width = line[index] === "\t" ? tabSize - (column % tabSize) : 1;
    if (targetColumn < column + width) return index;
    column += width;
  }
  return line.length;
}

function applyEditorChange(editor, lines, highlight) {
  state.code = editor.value;
  updateEditorDecorations(editor, lines, highlight);
  scheduleSave();
}

function selectedLineBounds(text, selectionStart, selectionEnd) {
  const start = text.lastIndexOf("\n", selectionStart - 1) + 1;
  let effectiveEnd = selectionEnd;
  if (selectionEnd > selectionStart && text[selectionEnd - 1] === "\n") {
    effectiveEnd -= 1;
  }
  const nextLineBreak = text.indexOf("\n", effectiveEnd);
  const end = nextLineBreak === -1 ? text.length : nextLineBreak;
  return { start, end };
}

function toggleLineComments(editor, lines, highlight) {
  const text = editor.value;
  const selectionStart = editor.selectionStart;
  const selectionEnd = editor.selectionEnd;
  const hasSelection = selectionStart !== selectionEnd;
  const bounds = selectedLineBounds(text, selectionStart, selectionEnd);
  const block = text.slice(bounds.start, bounds.end);
  const originalLines = block.split("\n");
  const nonBlankLines = originalLines.filter((line) => line.trim().length > 0);
  const shouldUncomment =
    nonBlankLines.length > 0 && nonBlankLines.every((line) => /^\s*# ?/.test(line));
  const transforms = originalLines.map((line) => transformCommentLine(line, shouldUncomment));
  const replacement = transforms.map((item) => item.changed).join("\n");

  editor.value = `${text.slice(0, bounds.start)}${replacement}${text.slice(bounds.end)}`;

  if (hasSelection) {
    editor.setSelectionRange(bounds.start, bounds.start + replacement.length);
  } else {
    const offset = selectionStart - bounds.start;
    const nextOffset = mapLineTransformOffset(transforms, offset);
    const nextSelection = bounds.start + nextOffset;
    editor.setSelectionRange(nextSelection, nextSelection);
  }

  applyEditorChange(editor, lines, highlight);
}

function transformCommentLine(line, shouldUncomment) {
  if (shouldUncomment) {
    const match = line.match(/^(\s*)# ?/);
    if (!match) {
      return {
        changed: line,
        column: 0,
        delta: 0,
        removeLength: 0,
        originalLength: line.length,
      };
    }
    const column = match[1].length;
    const removeLength = match[0].length - column;
    return {
      changed: `${line.slice(0, column)}${line.slice(column + removeLength)}`,
      column,
      delta: -removeLength,
      removeLength,
      originalLength: line.length,
    };
  }

  const indent = line.match(/^\s*/)[0];
  const column = indent.length;
  return {
    changed: `${line.slice(0, column)}# ${line.slice(column)}`,
    column,
    delta: 2,
    removeLength: 0,
    originalLength: line.length,
  };
}

function mapLineTransformOffset(transforms, offset) {
  let originalCursor = 0;
  let nextCursor = 0;

  for (const transform of transforms) {
    const lineEnd = originalCursor + transform.originalLength;
    if (offset <= lineEnd) {
      return nextCursor + mapSingleLineOffset(transform, offset - originalCursor);
    }
    originalCursor = lineEnd + 1;
    nextCursor += transform.changed.length + 1;
  }

  return nextCursor;
}

function mapSingleLineOffset(transform, offset) {
  if (transform.delta > 0) {
    return offset >= transform.column ? offset + transform.delta : offset;
  }
  if (transform.delta < 0) {
    const removeEnd = transform.column + transform.removeLength;
    if (offset <= transform.column) return offset;
    if (offset <= removeEnd) return transform.column;
    return offset + transform.delta;
  }
  return offset;
}

function removeCurrentLine(editor, lines, highlight) {
  const text = editor.value;
  const cursor = editor.selectionStart;
  const lineStart = text.lastIndexOf("\n", cursor - 1) + 1;
  const lineBreak = text.indexOf("\n", cursor);
  let removeStart = lineStart;
  let removeEnd = text.length;
  let nextCursor = lineStart;

  if (lineBreak !== -1) {
    removeEnd = lineBreak + 1;
  } else if (lineStart > 0) {
    removeStart = lineStart - 1;
    nextCursor = removeStart;
  }

  const removed = text.slice(removeStart, removeEnd);
  editor.value = `${text.slice(0, removeStart)}${text.slice(removeEnd)}`;
  editor.setSelectionRange(nextCursor, nextCursor);
  writeClipboard(removed);
  applyEditorChange(editor, lines, highlight);
}

function writeClipboard(text) {
  if (!text || typeof navigator === "undefined" || !navigator.clipboard?.writeText) return;
  navigator.clipboard.writeText(text).catch(() => {});
}

function bindPaneResizer() {
  const handle = document.querySelector("[data-pane-resizer]");
  const workspace = document.querySelector(".workspace");
  if (!handle || !workspace) return;

  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    document.body.classList.add("pane-resizing");

    const workspaceRect = workspace.getBoundingClientRect();
    const workspaceStyle = window.getComputedStyle(workspace);
    const paddingLeft = Number.parseFloat(workspaceStyle.paddingLeft) || 0;
    const paddingRight = Number.parseFloat(workspaceStyle.paddingRight) || 0;
    const contentLeft = workspaceRect.left + paddingLeft;
    const contentWidth = workspaceRect.width - paddingLeft - paddingRight;
    const handleWidth = handle.getBoundingClientRect().width;
    const minProblemWidth = Math.min(320, contentWidth * 0.45);
    const minIdeWidth = Math.min(420, contentWidth * 0.45);
    const maxProblemWidth = Math.max(
      minProblemWidth,
      contentWidth - minIdeWidth - handleWidth,
    );

    const applyWidth = (clientX) => {
      if (contentWidth <= 0) return;
      const nextWidth = boundedNumber(
        clientX - contentLeft,
        minProblemWidth,
        minProblemWidth,
        maxProblemWidth,
      );
      const nextPct = Math.round((nextWidth / contentWidth) * 1000) / 10;
      state.problemPaneWidthPct = nextPct;
      workspace.style.setProperty("--problem-pane-width", `${nextPct}%`);
      localStorage.setItem("leetgpu.problemPaneWidthPct", String(nextPct));
    };

    const onMove = (moveEvent) => {
      applyWidth(moveEvent.clientX);
    };

    const onUp = () => {
      document.body.classList.remove("pane-resizing");
      if (handle.hasPointerCapture(event.pointerId)) {
        handle.releasePointerCapture(event.pointerId);
      }
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
    };

    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  });
}

function bindConsoleResizer() {
  const handle = document.querySelector("[data-console-resizer]");
  const pane = document.querySelector(".right-pane");
  if (!handle || !pane) return;

  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    const startY = event.clientY;
    const startHeight = state.consoleHeight;
    const paneRect = pane.getBoundingClientRect();
    const minHeight = 120;
    const maxHeight = Math.max(180, paneRect.height - 160);

    const onMove = (moveEvent) => {
      const nextHeight = Math.min(maxHeight, Math.max(minHeight, startHeight - (moveEvent.clientY - startY)));
      state.consoleHeight = Math.round(nextHeight);
      pane.style.setProperty("--console-height", `${state.consoleHeight}px`);
      localStorage.setItem("leetgpu.consoleHeight", String(state.consoleHeight));
    };

    const onUp = () => {
      handle.removeEventListener("pointermove", onMove);
      handle.removeEventListener("pointerup", onUp);
      handle.removeEventListener("pointercancel", onUp);
    };

    handle.addEventListener("pointermove", onMove);
    handle.addEventListener("pointerup", onUp);
    handle.addEventListener("pointercancel", onUp);
  });
}

function updateEditorDecorations(editor, lines, highlight) {
  const count = Math.max(1, editor.value.split("\n").length);
  let text = "";
  for (let index = 1; index <= count; index += 1) {
    text += `${index}\n`;
  }
  lines.textContent = text;
  highlight.innerHTML = highlightPython(
    editor.value,
    analyzePythonBrackets(editor.value, activeBracketIndex(editor)),
  );
  syncEditorScroll(editor, lines, highlight);
}

function syncEditorScroll(editor, lines, highlight) {
  lines.scrollTop = editor.scrollTop;
  highlight.scrollTop = editor.scrollTop;
  highlight.scrollLeft = editor.scrollLeft;
}

function scheduleSave() {
  const saveState = document.getElementById("saveState");
  if (saveState) saveState.textContent = "Saving";
  window.clearTimeout(state.saveTimer);
  state.saveTimer = window.setTimeout(async () => {
    if (!state.current) return;
    try {
      await api("/api/save", {
        method: "POST",
        body: JSON.stringify({ challengeId: state.current.id, code: state.code }),
      });
      const target = document.getElementById("saveState");
      if (target) target.textContent = "Saved";
    } catch {
      const target = document.getElementById("saveState");
      if (target) target.textContent = "Save failed";
    }
  }, 450);
}

async function judge(action) {
  if (!state.current || state.running) return;
  const editor = document.getElementById("codeEditor");
  if (editor) state.code = editor.value;
  const selectedDevice = document.querySelector("[data-device]")?.value || state.device || "auto";
  const submittedChallengeId = state.current.id;
  let rendered = false;
  state.device = selectedDevice;
  state.running = true;
  renderChallenge();
  try {
    const result = await api("/api/judge", {
      method: "POST",
      body: JSON.stringify({
        challengeId: state.current.id,
        code: state.code,
        action,
        device: selectedDevice,
      }),
    });
    state.lastResult = result;
    await loadChallenges();
    const nextChallenge =
      action === "submit" && result.status === "passed"
        ? nextUnsolvedChallenge(submittedChallengeId)
        : null;
    state.current = await api(`/api/challenge?id=${encodeURIComponent(submittedChallengeId)}`);
    state.current.code = state.code;
    state.running = false;
    renderChallenge();
    rendered = true;
    if (action === "submit" && result.status === "passed") {
      promptNextUnsolved(nextChallenge);
    }
  } catch (error) {
    state.lastResult = {
      success: false,
      status: "error",
      message: error.message,
      summary: { passed: 0, failed: 1, skipped: 0 },
      tests: [],
      durationMs: 0,
    };
  } finally {
    if (!rendered) {
      state.running = false;
      renderChallenge();
    }
  }
}

function promptNextUnsolved(nextChallenge) {
  closeNextChallengePrompt();
  if (!nextChallenge) {
    showNextChallengePrompt({
      title: "Submission Passed",
      message: "No unsolved challenges remain.",
      primaryLabel: "OK",
      onPrimary: closeNextChallengePrompt,
    });
    return;
  }

  showNextChallengePrompt({
    title: "Submission Passed",
    message: `Go to the next unsolved challenge?\n${nextChallenge.title}`,
    primaryLabel: "Yes",
    secondaryLabel: "No",
    onPrimary: () => {
      closeNextChallengePrompt();
      state.detailTab = "problem";
      window.location.hash = challengeHref(nextChallenge.id);
    },
  });
}

function showNextChallengePrompt({ title, message, primaryLabel, secondaryLabel = "", onPrimary }) {
  const modal = document.createElement("div");
  modal.className = "modal-backdrop";
  modal.dataset.nextChallengeModal = "true";
  modal.innerHTML = `
    <section class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="nextChallengeTitle">
      <h2 id="nextChallengeTitle">${escapeHtml(title)}</h2>
      <p>${escapeHtml(message)}</p>
      <div class="modal-actions">
        ${secondaryLabel ? `<button class="ghost-button" data-modal-close>${escapeHtml(secondaryLabel)}</button>` : ""}
        <button class="primary-button green" data-modal-primary>${escapeHtml(primaryLabel)}</button>
      </div>
    </section>
  `;
  document.body.appendChild(modal);

  const primaryButton = modal.querySelector("[data-modal-primary]");
  const closeButton = modal.querySelector("[data-modal-close]");
  primaryButton?.addEventListener("click", onPrimary);
  closeButton?.addEventListener("click", closeNextChallengePrompt);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) closeNextChallengePrompt();
  });
  modal.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeNextChallengePrompt();
  });
  primaryButton?.focus();
}

function closeNextChallengePrompt() {
  document.querySelector("[data-next-challenge-modal]")?.remove();
}

async function refreshData() {
  await Promise.all([loadRuntime(), loadChallenges()]);
  if (state.homeTab === "submissions") await loadGlobalSubmissions();
}

function typesetMath() {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([app]).catch(() => {});
  }
}

async function boot() {
  renderLoading("Loading challenges...");
  try {
    await refreshData();
    window.addEventListener("hashchange", route);
    route();
  } catch (error) {
    renderError(error.message);
  }
}

boot();
