# Adjudication of the 36 test-fold disagreements

- Adjudicator: Iqra Iqbal Khan (consented to be named). Date completed: 2026-11-12.
- Confirmed by the project team: the adjudicator did not annotate any of these posts and had no role in building or running the models.
- Material supplied: a worksheet (post text, indicator, indicator definition), annotation guidelines v1.1 FINAL and the v1.1 addendum. Annotator labels, released labels and model outputs were not shown.
- Scope: 36 post-indicator disagreements between Annotator 1 and Annotator 2 on 29 test-fold posts (5 canonical, 11 mined, 20 re-annotated). All other test cells keep their released labels.
- Decisions: 22 Present, 14 Absent, 0 Indeterminate. Sided with Annotator 1 in 16 cases and Annotator 2 in 20.
- Consistency with the earlier canonical adjudication (Sadia Ramzan): 3 of 5 shared cases agree; C28 (mobilization) and C30 (threat) differ.
- Adjudicator profile: native Urdu speaker, proficient in Roman Urdu, MS in Computer Science, no relationship with the research team.
- One adjudicator only; reasons were written one line per row.
- File: `adjudication_36_decisions_no_text.csv` (columns a1, a2 = the two annotators' labels; decision; side; reason; adjudicator; date). The worksheet with post text is not released.
- The released labels in `data/` are unchanged. The adjudicated labels are used only for the sensitivity analysis (`results/macroF1_by_reference_incl_adjudicated.csv`, `results/paired_bootstrap_change_in_gap.csv`).
