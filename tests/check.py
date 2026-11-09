import torch


def assert_close(name, mine, ref, *, atol=1e-5, rtol=1e-5, hint=None):
    """Compare your tensor to the reference, failing with a message that says what is off."""
    tail = f"\n  hint: {hint}" if hint else ""
    assert isinstance(mine, torch.Tensor), f"{name}: returned {type(mine).__name__}, expected a Tensor{tail}"
    assert mine.shape == ref.shape, f"{name}: shape {tuple(mine.shape)}, expected {tuple(ref.shape)}{tail}"
    assert mine.dtype == ref.dtype, f"{name}: dtype {mine.dtype}, expected {ref.dtype}{tail}"
    assert mine.device == ref.device, f"{name}: on {mine.device}, expected {ref.device}{tail}"
    bad = ~torch.isfinite(mine)
    if bad.any():
        first = tuple(bad.nonzero()[0].tolist())
        raise AssertionError(
            f"{name}: {int(bad.sum())} non-finite values (NaN/inf), first at index {first}"
            f" = {mine[first].item()}{tail}"
        )
    try:
        torch.testing.assert_close(mine, ref, atol=atol, rtol=rtol)
    except AssertionError as e:
        raise AssertionError(f"{name}: values differ from reference.\n{e}{tail}") from None
