# AutoInsight 🔍

**An autonomous multi-agent system that turns a raw CSV and a plain-English business question into a statistically sound, human-reviewed data analysis report — with zero manual coding.**

Built as a deep dive into agentic AI system design: LangGraph orchestration, human-in-the-loop control, sandboxed code execution, and statistically rigorous automated reasoning.

---

## What it does

Give AutoInsight a dataset and a question like *"Is there a significant difference in salary between departments?"* — it will:

1. **Profile** the data (types, missing values, skewness, normality — via Shapiro-Wilk testing)
2. **Route** to the statistically correct methodology using a hybrid rule engine + semantic search over 60+ statistical test descriptions
3. **Plan** a multi-step analysis strategy using an LLM, constrained by hard rules (e.g. cleaning must come first if data is dirty)
4. **Pause for human approval** of the plan before executing anything
5. **Generate and run Python code** for each step inside an isolated, AST-security-scanned Docker sandbox
6. **Self-correct**: a Critic agent checks for statistical assumption violations (e.g. running a t-test on non-normal data) and execution errors, sending code back for revision — with bounded retries and graceful degradation
7. **Report** findings in business language, with confidence levels per finding and explicit limitations for anything that didn't fully pass validation

The output isn't just "an LLM wrote some code" — every number in the final report is traced back to verified execution output, and every methodology choice is checked against the data's actual statistical properties before it's used.

---

## Why this project

Most "AI data analyst" demos let an LLM freely write and run code, then narrate whatever comes out. That's fast, but it's also how you get a t-test confidently applied to wildly skewed data, or a report that states a p-value the code never actually produced.

AutoInsight was built to explore what it takes to make an agentic system **trustworthy**, not just functional:

- Statistical validity is enforced structurally, not left to the LLM's judgment
- A human stays in the loop at the one point where oversight matters most — before any code runs
- Code execution is sandboxed and security-scanned (AST-level import/call allowlisting), not just "trust the LLM's output"
- Every claim in the final report is grounded in metrics that were actually computed, tested against hallucination via automated evals

---

## Architecture

A 7-node LangGraph state machine, with a conditional routing loop for retries:

```
 Profiler → Router → Planner → [human approval] → Coder → Sandbox → Critic
                                                       ↑              │
                                                       └──────────────┘
                                                     (retry on failure,
                                                      up to max_retries)
                                                              │
                                                              ▼
                                                          Reporter
```

| Node | Role |
|---|---|
| **Profiler** | Polars-based statistical profiling — dtypes, missingness, cardinality, skewness, Shapiro-Wilk normality tests |
| **Router** | Hybrid methodology selection: fast rule-based JSON matching first, falls back to ChromaDB semantic search over 60+ statistical method descriptions |
| **Planner** | LLM (Groq/Llama 3.3 70B) generates a 3–6 step analysis plan via structured tool-calling, constrained by hard rules derived from the data profile |
| **Coder** | LLM generates Polars/SciPy code per step, using only an approved API surface, with prior error context injected on retries |
| **Sandbox** | Executes code in an isolated Docker container via Jupyter kernel protocol, after AST-level security scanning (blocks `os`, `subprocess`, `eval`, dangerous dunder access, etc.) |
| **Critic** | Validates execution success *and* statistical assumption correctness (e.g. flags a parametric test run on non-normal data and demands the non-parametric equivalent); bounded retries with graceful degradation to "low confidence" rather than infinite loops |
| **Reporter** | LLM writes the final business-facing report strictly from verified metrics — no invented numbers, explicit confidence levels and limitations per finding |

**Stack:** LangGraph · Polars · ChromaDB · FastAPI · Streamlit · Groq (Llama 3.3 70B) · Docker · Jupyter kernel protocol · LangSmith (tracing) · DeepEval (automated testing) · SciPy / statsmodels / scikit-posthocs

---

## Engineering highlights

- **Statistical guardrails, not just prompting.** The Critic node hard-codes the mapping between parametric tests and their non-parametric equivalents (t-test → Mann-Whitney U, ANOVA → Kruskal-Wallis + Dunn post-hoc, Pearson → Spearman) and cross-checks every methodology choice against the data's actual Shapiro-Wilk results before accepting it.
- **Security-scanned code execution.** Generated code is parsed into an AST and checked against an import/function allowlist before ever reaching the sandbox — blocking `os`, `subprocess`, `eval`, `__globals__`/`__subclasses__` traversal, and other sandbox-escape patterns, independent of what the LLM was told not to do.
- **Bounded self-correction.** Failed or statistically invalid code goes back to the Coder with the exact error and Critic feedback injected into the next prompt — capped at a configurable retry limit, after which the step is marked low-confidence rather than looping forever or silently failing.
- **Grounded reporting.** The Reporter is evaluated with DeepEval's faithfulness metrics to catch cases where the LLM's narrative drifts from the actual computed metrics — a structural test suite (16+ tests) also directly validates router intent-detection, non-parametric fallback logic, and critic violation-detection deterministically, without depending on LLM output at all.
- **Human-in-the-loop by design.** The graph is compiled with an explicit interrupt before code execution begins — the analysis plan is always shown to a human for approval first, rather than an agent silently running arbitrary generated code against real data.

---

## Status

Core pipeline (profiling → routing → planning → human approval → coding → sandboxed execution → critique/retry → reporting) is fully implemented and tested end-to-end locally, with an automated test suite covering router, critic, and reporter correctness.
