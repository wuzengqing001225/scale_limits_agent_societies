# Predicting the scale limits of social mechanisms in agent societies

Code and data for "Predicting the scale limits of social mechanisms in agent societies".

The audit predicts, before results are known, which effects of a social
mechanism survive population scaling, by separating (i) the mechanism's
structural quantities, (ii) the information its agents actually consume,
and (iii) the observation protocol.

## Layout

```
code/                 all scripts, flat; every script assumes ../data
data/                 all experiment outputs (JSON summaries + JSONL raw call
                      logs, one row per LLM decision with resolved model id)
data/pr_extract/      field study: per-venue reconciliation counts, status
                      breakdowns and provenance logs (ICLR 2024-2026, TMLR),
                      the 2025 area hold-out split, and the frozen-protocol
                      analysis outputs
data/archive/         superseded raw logs retained for audit (rate-limit and
                      misconfigured-engine batches; excluded from analyses)
predictions_ledger.md digest of every prediction and adjudication criterion
                      logged before the corresponding experiment ran, with
                      outcomes and data pointers
```

## Reproducing key results

| Result (paper) | Script | Data |
|---|---|---|
| Failure-scale product law, exponent 1.00 | `r3_prime.py` | `r3_prime_*` |
| Paired experiments and interventions | `r1_*`, `r4_*`, `r5_*`, `anchors_data.py` | matching `*_results.json` |
| Coverage/institution flip | `r6_board.py` | `r6_results.json` |
| Red team, transfer, prospective test | `a1_*`, `t1_*`, `r7_*` | matching results files |
| Count-format lattice (beta_logN = 0.00) | `p6_dose.py` | `p6_main_raw.jsonl` |
| Cross-architecture count lattices | `p6c_count_grid.py` | `p6c_*` |
| Percentage lattices and GLM/Firth fits | `div1b_grid.py`, `div_glm.py` | `div1b_*`, `div_glm_results.json` |
| Sealed cross-architecture test | `div1_divergence.py` | `div1_*` |
| Baseline probes, rewordings, cross-model | `probe_*.py`, `p4f_p6b.py`, `p5c_crossarch.py` | matching raw/summary files |
| Temperature sweep | `t_sens.py` | `t_sens_gemini_*` (and the voided primary-engine arm `t_sens_raw.jsonl`, kept because the provider rejects non-default temperature) |
| Composed system, zero ignition | `p3_composition.py` | `p3_cache.jsonl` |
| Field study (clustering, TMLR contrast, S2) | `pr_extract.py`, `pr_analysis.py` | `data/pr_extract/` |
| Device accumulation 5 to 39 | coding matrix | `h4_devices_en.csv` |
| SI tables (regenerated, not hand-typed) | `make_si_tables.py` | summary files above |
| All figures | `make_figures.py` | as above |

Analysis scripts (`div_glm.py`, `pr_analysis.py`, `make_si_tables.py`,
`make_figures.py`) run offline from the released data. Probe scripts call
provider CLIs/APIs and reproduce protocols rather than bit-identical
outputs; directional findings replicated across snapshots in our runs,
while levels are snapshot-bound (see paper).

## Field data policy

Raw OpenReview review and edit corpora are not redistributed (full review
texts). `pr_extract.py` re-fetches them from the public API under its terms
(authenticated access, at most 2 requests/s, resume-safe, provenance
logged). The released reconciliation counts and provenance logs document
the extraction, and derived analysis outputs are included.

## Requirements

Python >= 3.10; see `requirements.txt`. Rule-based experiments are fully
deterministic under their explicit seeds.
