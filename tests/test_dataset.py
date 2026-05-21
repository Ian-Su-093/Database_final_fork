"""Tests for PRMDataset."""
import os, sys, json
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'clause_ppo', 'data', 'processed')

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(PROCESSED_DIR, 'corruption_dataset.json')),
    reason="Run build_corruption_dataset.py --max_examples 100 first"
)


def _make_ds(max_length=256):
    from data.dataset import PRMDataset
    return PRMDataset(
        PROCESSED_DIR,
        tokenizer_name='gpt2',
        max_length=max_length,
        schema_dropout_prob=0.0,
    )


def test_dataset_loads_without_error():
    ds = _make_ds()
    assert len(ds) > 0


def test_item_has_required_keys():
    ds = _make_ds()
    item = ds[0]
    assert 'input_ids'      in item
    assert 'attention_mask' in item
    assert 'label'          in item


def test_item_shapes_and_types():
    ds = _make_ds()
    item = ds[0]
    assert item['input_ids'].dtype      == torch.long
    assert item['attention_mask'].dtype == torch.long
    assert item['label'].dtype          == torch.float
    assert item['label'].shape          == torch.Size([])


def test_label_range():
    ds = _make_ds()
    for i in range(min(50, len(ds))):
        label = ds[i]['label'].item()
        assert label in (0.0, 1.0), f"Label {label} is not 0.0 or 1.0"


def test_sequence_padded_to_max_length():
    ds = _make_ds(max_length=128)
    item = ds[0]
    assert item['input_ids'].shape[0] == 128
