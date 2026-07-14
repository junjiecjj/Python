# Stage 4 ZF-SNR Gain vs N_RIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dense Monte Carlo `N_RIS` sweep for random and fixed-grid optimized ZF output SNR gains.

**Architecture:** Add one standalone MATLAB main function that sweeps `N_RIS`, regenerates dimension-consistent channels and RIS phases per trial, evaluates random and fixed-grid optimized ZF-SNR metrics, prints progress, and saves statistics plus publication-ready diagnostic figures. Keep the existing RD and Pd-vs-SNR scripts untouched.

**Tech Stack:** MATLAB scripts/functions, existing channel generator, objective evaluator, fixed-grid optimizer, MATLAB unittest.

---

### Task 1: Smoke Test

**Files:**
- Create: `tests/test_stage4_snr_gain_vs_nris.m`

- [ ] Write a reduced-axis smoke test that calls the new main function with `NrisAxis=[4,8]` and a tiny trial count.
- [ ] Verify result axes, metric matrix sizes, runtime, and saved-output toggle fields.
- [ ] Run the test and confirm it fails before the main function exists.

### Task 2: Dense N_RIS Sweep

**Files:**
- Create: `main/main_stage4_snr_gain_vs_nris.m`

- [ ] Implement dense default config `4:4:64`, `100` trials, fixed-grid optimizer settings, per-trial progress printing, Monte Carlo metric storage, summaries, `.mat`, `.png`, and `.txt` output.
- [ ] Keep channel dimensions self-consistent for each `N_RIS`.
- [ ] Run the smoke test until green.

### Task 3: Local Validation

**Files:**
- Run: `main/main_stage4_snr_gain_vs_nris.m`

- [ ] Execute a small override run locally to verify saved artifacts and plots without completing the full default sweep.
- [ ] Inspect the plotted SNR/gain/runtime panels.

### Task 4: Chinese Docs

**Files:**
- Modify: `docs/project_architecture.md`
- Modify: `docs/paper_formula_notes.md`
- Modify: `docs/reproduction_assumptions.md`
- Modify: `docs/experiment_log.md`
- Modify: `docs/todo.md`
- Modify: `docs/debug_log.md` only when a debug issue occurs

- [ ] Record that this experiment replaces Pd as the primary `N_RIS` hardware trend metric with ZF output SNR and gain curves.
