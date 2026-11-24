import pytest
import media.bitdepth_dither as mod

CASES = [
    ("seed_per_track", (1,)),
    ("coeff_by_rate", (1,)),
    ("skip_silence", (1,)),
    ("carry_state_across_segments", (1,)),
    ("dither_before_limit", (1,)),
    ("clamp_peak", (1,)),
    ("convert", (1,)),
    ("noise_shape", (1,)),
    ("requantize", (1,)),
    ("export", (1,)),
    ("validate_dither", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
