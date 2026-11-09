import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is"

torch.set_grad_enabled(False)

# Built-ins that would let a lesson's exercise skip the point of the exercise.
_BANNED = {
    "softmax": [
        (torch, "softmax"),
        (torch, "log_softmax"),
        (torch.special, "softmax"),
        (torch.nn.functional, "softmax"),
        (torch.nn.functional, "log_softmax"),
        (torch.Tensor, "softmax"),
        (torch.Tensor, "log_softmax"),
    ],
    "rms_norm": [(torch, "rms_norm"), (torch.nn.functional, "rms_norm")],
    "embedding": [(torch, "embedding"), (torch.nn.functional, "embedding")],
    "linear": [(torch.nn.functional, "linear")],
}


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def hf(device):
    """The reference: Hugging Face's own Qwen3-0.6B, in fp32."""
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()


@pytest.fixture(scope="session")
def W(hf):
    """HF weights by name. Shares storage with `hf`, so no extra VRAM."""
    return dict(hf.state_dict())


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.fixture(scope="session")
def ids(tok, device):
    return tok(PROMPT, return_tensors="pt").input_ids[0].to(device)


@pytest.fixture
def forbid(monkeypatch):
    """Call after computing the reference: makes the named built-ins raise."""

    def _forbid(*groups: str):
        for group in groups:
            for owner, name in _BANNED[group]:
                if not hasattr(owner, name):
                    continue

                def banned(*_a, _name=name, **_k):
                    raise AssertionError(f"`{_name}` is off limits in this exercise: build it from tensor ops.")

                monkeypatch.setattr(owner, name, banned)

    return _forbid
