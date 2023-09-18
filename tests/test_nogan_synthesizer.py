"""Tests for `nogan_synthesizer` package."""

import numpy as np
import pandas as pd
import pytest
from nogan_synthesizer import NoGANSynth


# Define test cases
def test_nogan_synth():
    # Column Names with spaces and special characters
    real_data = pd.DataFrame(np.random.rand(10000, 5), columns=["x@1", "x'2", "xअ3", "x4", "x 5"])
    
    nogan = NoGANSynth(real_data, random_seed = 42)
    nogan.fit()
    
    n_synth_rows = len(synth_data)
    synth_data = nogan.generate_synthetic_data(no_of_rows=n_synth_rows)
    
    assert len(synth_data.columns) == 5 and synth_data.columns.isalnum() == True

    
