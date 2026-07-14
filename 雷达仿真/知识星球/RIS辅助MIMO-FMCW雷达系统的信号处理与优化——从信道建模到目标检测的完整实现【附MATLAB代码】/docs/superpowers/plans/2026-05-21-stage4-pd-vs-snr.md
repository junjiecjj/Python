# Stage 4 Pd-vs-SNR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a quick/full Monte Carlo Stage 4 experiment for CA-CFAR detection probability versus echo SNR.

**Architecture:** Keep the existing Stage 4 four-target FMCW and CFAR functions intact. Add one MATLAB main script that builds the RIS baselines once, converts an optimized-reference echo SNR axis to echo noise power, performs Monte Carlo noise resampling, accumulates per-target CFAR hits, prints progress, and saves plots/data/logs.

**Tech Stack:** MATLAB scripts/functions, existing FMCW/RD/CFAR functions, MATLAB unittest.

---

### Task 1: Quick Acceptance Test

**Files:**
- Create: `tests/test_stage4_pd_vs_snr.m`
- Test: `tests/test_stage4_pd_vs_snr.m`

- [ ] Write a failing smoke test that runs `main_stage4_pd_vs_snr("quick")` with a small override and verifies output fields for SNR axis, per-target Pd, average Pd, and saved artifacts.
- [ ] Run the test and confirm it fails because the new main script does not exist.

### Task 2: Pd-vs-SNR Main Script

**Files:**
- Create: `main/main_stage4_pd_vs_snr.m`
- Test: `tests/test_stage4_pd_vs_snr.m`

- [ ] Add quick/full configuration, four-target setup, RIS baselines, echo-SNR-to-noise conversion, CA-CFAR Monte Carlo loop, per-trial `fprintf`, accumulation, figure export, MAT export, and log export.
- [ ] Keep existing Stage 4 main flow unchanged.
- [ ] Run the smoke test and confirm it passes.

### Task 3: Quick Experiment Verification

**Files:**
- Run: `main/main_stage4_pd_vs_snr.m`

- [ ] Run quick mode with its normal trial count.
- [ ] Check saved MAT, PNG, TXT outputs.
- [ ] Confirm quick trends: optimized RIS average Pd exceeds random RIS, No RIS stays lowest, and average Pd rises overall with SNR.

### Task 4: Chinese Project Records

**Files:**
- Modify: `docs/project_architecture.md`
- Modify: `docs/paper_formula_notes.md`
- Modify: `docs/reproduction_assumptions.md`
- Modify: `docs/experiment_log.md`
- Modify: `docs/debug_log.md` if a debug issue is encountered
- Modify: `docs/todo.md`

- [ ] Record Pd-vs-SNR experiment scope, SNR reference definition, CFAR detection rule, quick result, artifacts, and full-mode follow-up.
