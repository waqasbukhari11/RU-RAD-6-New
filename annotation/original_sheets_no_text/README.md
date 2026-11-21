Per-annotator labels (six indicators) with `uncertain` and, where recorded, `notes`. Post text is removed.
Notes are the annotators' own comments; a few quote one to three words of a post to explain a decision.

- canonical350_annotator{1,2}: the 350 double-annotated posts (batch `iaa`) plus that annotator's solo half (675 rows each).
- mining320_annotator{1,2}: the 320 mined posts (batch `mined_320`).
- reannotated200_annotator{1,2}: the 200-post batch (`solo2_200_relabelled`). Annotator 2's sheet is the
  re-annotation made after the correction note in `guidelines/annotator_correction_note.md`.
  The two sheets differ on 82 post-indicator decisions; the released label is positive only where both are positive.
