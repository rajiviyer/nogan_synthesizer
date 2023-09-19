# NOGAN SYNTHESIZER
<!-- [![PyPI version](https://badge.fury.io/py/genai-evaluation.svg)](https://badge.fury.io/py/genai-evaluation) -->
[![Documentation](https://img.shields.io/badge/Documentation-%20-blue)](https://rajiviyer.github.io/nogan_synthesizer/)


NoGANSynthesizer is a library which generates synthetic tabular data based on methods of multivariate binning. It offers faster, more accurate and less complex alternative to GAN. 

## Class
- **NoGANSynthesizer**: Synthetic Data Generator that fits a tabular data

## Functions
- **wrap_category_columns**: Function to compress all specified categorical columns into one
- **unwrap_category_columns**: Function to expand all wrapped categorical columns

## Authors
- [Dr. Vincent Granville](mailto:vincentg@mltechniques.com) - Research
- [Rajiv Iyer](mailto:raju.rgi@gmail.com) - Development/Maintenance

## Installation
The package can be installed with
```
pip install nogan_synthesizer
```

## Tests
The test can be run by cloning the repo and running:
```
pytest tests
```
In case of any issues running the tests, please run them after installing the package locally:

```
pip install -e .
```

## Usage

Start by importing the class
```Python
from nogan_synthesizer import NoGANSynth
from nogan_synthesizer.preprocessing import wrap_category_columns, unwrap_category_columns
from genai_evaluation import multivariate_ecdf, ks_statistic
```

Assuming we have a pandas dataframe (Real) having some categorical columns and we are interested in generating Synthetic based on that.
We first prepocess the categorical columns which will return preprocessed real dataset & its corresponding flag vector index to key value dictionary
```Python
cat_cols = [category columns list...]
wrapped_real_data, idx_to_key, key_to_idx = \
                        wrap_category_columns(real_data, cat_cols)
```

We then fit the NoGANSynth Model on the wrapped dataset and generate synthetic data
```Python
nogan = NoGANSynth(real_data)
nogan.fit()

n_synth_rows = len(real_data)
synth_data = nogan.generate_synthetic_data(no_of_rows=n_synth_rows)
```

We can then evaluate the synthetic & real data distributions using genai_evaluation package
```Python
_, ecdf_val1, ecdf_synth = \
            multivariate_ecdf(wrapped_real_data, 
                              synth_data, 
                              n_nodes = 1000,
                              verbose = True,
                              random_seed=42)

ks_stat = ks_statistic(ecdf_val1, ecdf_synth)                              
```

Once we are satisfied with the evaluation results, we can unwrap the Generated Synthetic dataset (unwrap the categorical columns) using the previously generated flag vector index to key dictionary
```Python
unwrapped_synth_data = unwrap_category_columns(synth_data, idx_to_key, cat_cols)
```
## Motivation
The motivation for this package comes from Dr. Vincent Granville's paper [Generative AI Technology Break-through: Spectacular Performance of New Synthesizer](https://mltechniques.com/2023/08/02/generative-ai-technology-break-through-spectacular-performance-of-new-synthesizer/)

If you have any tips or suggestions, please contact us on email.