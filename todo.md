# TODO

## 1. Metadata filter panel for the grid

Add a panel to explicitly control which images are eligible for the grid, using
criteria on **any column of the `LabelChecker*.csv`** (e.g. a manual filter like
`Length > 0.5`, `AbdArea >= 100`, `LabelPredicted == "Detritus"`).

**Why:** in the early state the ImageNet-pretrained model knows nothing, so its
predictions can't usefully rank the pool. Letting the annotator narrow the pool
by known measurements (size, shape, intensity, predicted label, …) surfaces
plausible candidates for a class before the model is trained, bootstrapping the
first labels.

**Sketch / notes:**
- A filter builder in the Annotate tab: pick a column, an operator
  (`>`, `>=`, `<`, `<=`, `==`, `!=`, contains), and a value; combine a few with
  AND. Apply on top of the existing pool filter (unlabelled, x not excluded).
- The eligible column set comes from the `DatasetSource` (it owns the CSV), not
  hardcoded — expose available columns + dtypes so the UI can offer them.
- Applies to both grid modes and the active-learning queue's pool sampling.
- Persist the active filter in the `Session` (like target/mode) so it survives
  tab switches; consider saving named filters to `annotate_settings.json`.
- Show how many pool images pass the filter so the annotator sees its effect.
