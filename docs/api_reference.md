# API Reference

### NoGANSynth Class and Methods
NoGAN Tabular Data Synthesizer generates synthetic data based on the multivariate binning technique performed on the Training or Real Dataset.

::: nogan_synthesizer.NoGANSynth

### Preprocessing

**wrap_category_columns**(*data: pd.DataFrame, cat_cols: List[str]*)

Categorical Columns can be preprocessed using key-value pairs (called flag vector) of all categorical columns and collapsing all these columns into a single feature with integer values. **wrap_category_columns** implements this concept.

::: nogan_synthesizer.preprocessing.wrap_category_columns

-----------------------
**unwrap_category_columns**(*data: pd.DataFrame, idx_to_key: dict, cat_cols: List[str]*)

All the collapsed categorical columns can also be expanded using the same flag vector created during wrapping process

::: nogan_synthesizer.preprocessing.unwrap_category_columns