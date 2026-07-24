# Time-ordered internal prediction ledger (digest)

Predictions and adjudication criteria were logged internally before the
corresponding experiments ran. This digest lists the sealed items and their
outcomes; full adjudications map to data files in `data/`.

## Third-party prospective test (R7)
Frozen new parameter grid on unmodified axelrod-python. Filed before
execution: 20 quantities (4 payoff-advantage curves x 5 population sizes),
including a sign-flip window between N=20 and N=40 and a marginally
positive cell. Outcome: 20/20 exact (deterministic system).
Data: `data/rule_based/r7_*`.

## Transfer test (T1)
Registered sign-flip window N=20..40 on the same library, standard grid.
Outcome: pointwise hit including +0.07 marginal cell; one exploratory
observable reversed direction (logged as failure).

## Sealed cross-architecture test (DIV-1)
Filed before any data: graded arm (deepseek-reasoner) monotone rank
8<40<200<1000; anchors give<=0.2 at N=8 and >=0.8 at N=1000; interior
values 0.396 (N=40) and 0.682 (N=200) from a two-anchor logistic, bands
+/-0.20; rise >=0.4. Companion flatness arm (claude-sonnet-5): spreads
<0.15. Outcome: graded arm 6/6 pass (0.500 and 0.708 observed); flatness
arm 0/4, registered failure branch executed; failure explained by the
threshold response found in the percentage lattice.
Data: `data/probes/div1_*`.

## Count lattice cross-architecture criteria (P6c)
Per engine, filed before execution: (i) dose response significant and
positive; (ii) 95% CI of beta_logN contains 0; (iii) anti-diagonal
(equal-proportion) cells unequal, spread >0.15. Outcomes: gemini 3/3,
gpt-5.5 3/3, deepseek-reasoner 2/3 (dose gate failed; recorded as
consistent rather than confirmed). Data: `data/probes/p6c_*`.

## Field study sealed items
2025 corpus split by area (seed 20260719) before analysis; validation half
run once against three criteria set in advance: (A1) last-72h share >=
0.27 (= 2x uniform benchmark) — met (0.608); (A2) area-size/clustering
rank correlation positive — sign met (+0.18) but effect near zero and
sign-unstable across halves; the criterion itself is recorded as too
coarse and uniformity is reported as the finding; (A3) revision rate in
[0.22, 0.32] — met (0.282). Data: `data/field_study/`.

## Failed or inconclusive registered outcomes
Twelve entries; see Supplementary Information of the paper for the table.
