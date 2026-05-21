"""Tests for ClausePRM model."""
import os, sys
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA GPU"
)


def test_model_instantiates():
    from models.prm import ClausePRM
    model = ClausePRM(
        model_name='codellama/CodeLlama-7b-hf',
        lora_rank=4,
        lora_alpha=8,
        use_4bit=True,
    )
    assert model is not None


def test_forward_returns_scalar_in_01():
    from models.prm import ClausePRM
    from transformers import AutoTokenizer

    model = ClausePRM(
        model_name='codellama/CodeLlama-7b-hf',
        lora_rank=4,
        lora_alpha=8,
        use_4bit=True,
    )
    tok = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-hf', use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    enc = tok("SELECT count(*) FROM head", return_tensors='pt', padding='max_length',
               max_length=32, truncation=True)
    enc = {k: v.cuda() for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    assert out.shape == torch.Size([1])
    assert 0.0 <= out.item() <= 1.0


def test_trainable_parameters():
    from models.prm import ClausePRM
    model = ClausePRM(
        model_name='codellama/CodeLlama-7b-hf',
        lora_rank=4,
        lora_alpha=8,
        use_4bit=True,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")
    # With LoRA, trainable params should be < 5% of total
    assert trainable < 0.05 * total
