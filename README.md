# NOGAN SYNTHESIZER
<!-- [![PyPI version](https://badge.fury.io/py/genai-evaluation.svg)](https://badge.fury.io/py/genai-evaluation)
[![Documentation](https://img.shields.io/badge/Documentation-%20-blue)](https://rajiviyer.github.io/genai_evaluation/) -->


NoGANSynthesizer is a library which generates synthetic tabular data based on methods of multivariate binning. It offers faster, more accurate and less complex alternative to GAN. 

## Class
- **NoGANSynthesizer**: Synthetic Data Generator that fits a tabular data

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
```

Assuming we have two pandas dataframes (Real & Synthetic) and only numerical columns, we pass them to the multivariate_ecdf function which returns the computed multivariate ECDFs of both.
```Python
query_str, ecdf_real, ecdf_synth = multivariate_ecdf(real_data, synthetic_data, n_nodes = 1000, verbose = True)
```

We then calculate the multivariate KS Distance between the ECDFs
```Python
ks_stat = ks_statistic(ecdf_real, ecdf_synth)
```

## Motivation
The motivation for this package comes from Dr. Vincent Granville's paper [Generative AI Technology Break-through: Spectacular Performance of New Synthesizer](https://mltechniques.com/2023/08/02/generative-ai-technology-break-through-spectacular-performance-of-new-synthesizer/)

If you have any tips or suggestions, please contact us on email.