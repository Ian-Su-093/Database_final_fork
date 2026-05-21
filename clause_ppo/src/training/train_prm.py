"""
PRM training loop for CLAUSE-PPO Phase 1.

Implements:
  - BCE loss on scalar PRM output vs {0.0, 1.0} labels
  - Schema-grounding auxiliary loss: penalise high scores for prefixes that
    reference tokens absent from the schema string
  - Gradient accumulation (effective batch = batch_size * grad_accum_steps)
  - bfloat16 mixed precision via torch.autocast
  - Checkpoint saving every eval_every steps (best by score_gap)
  - JSONL training log

score_gap metric: mean(V_phi | correct prefix) - mean(V_phi | corrupt prefix).
This is used as a proxy for Localisation Accuracy during training.
"""

import json
import os
import re
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


def _schema_grounding_flag(text: str) -> float:
    """
    Return 1.0 if the prefix part of the text references a token not in the schema.
    Heuristic: extract identifier tokens from [PREFIX] span, check against [SCHEMA] span.
    Used to compute the schema-grounding auxiliary loss.
    """
    schema_start = text.find('[SCHEMA]')
    prefix_start = text.find('[PREFIX]')
    if schema_start == -1 or prefix_start == -1:
        return 0.0

    schema_str = text[schema_start + len('[SCHEMA]'): prefix_start]
    prefix_str = text[prefix_start + len('[PREFIX]'):]

    sql_keywords = {
        'select', 'from', 'where', 'group', 'by', 'having', 'order',
        'join', 'on', 'and', 'or', 'not', 'in', 'like', 'is', 'null',
        'distinct', 'as', 'count', 'sum', 'avg', 'max', 'min',
        'asc', 'desc', 'limit', 'between', 'exists',
    }
    schema_tokens = {t.lower() for t in re.findall(r'\b\w+\b', schema_str)}
    prefix_tokens = {
        t.lower() for t in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', prefix_str)
        if t.lower() not in sql_keywords
    }
    return 1.0 if (prefix_tokens - schema_tokens) else 0.0


def _compute_score_gap(model, loader, device) -> tuple[float, float]:
    """
    Compute dev BCE loss and score_gap = mean(score|label=1) - mean(score|label=0).
    A positive score_gap means the model is correctly distinguishing correct from
    corrupted prefixes. Target: score_gap > 0.3 after training.
    """
    model.eval()
    total_bce, n_bce = 0.0, 0
    scores_pos, scores_neg = [], []
    bce_fn = nn.BCELoss(reduction='mean')

    with torch.no_grad():
        for batch in loader:
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                scores = model(ids, mask)
                loss   = bce_fn(scores, labels)
            total_bce += loss.item() * len(labels)
            n_bce     += len(labels)
            for s, l in zip(scores.cpu().tolist(), labels.cpu().tolist()):
                (scores_pos if l == 1.0 else scores_neg).append(s)

    dev_bce  = total_bce / n_bce if n_bce > 0 else float('inf')
    gap = (sum(scores_pos) / len(scores_pos) - sum(scores_neg) / len(scores_neg)
           if scores_pos and scores_neg else 0.0)
    return dev_bce, float(gap)


def train_prm(config: dict, spider_dir: str, processed_dir: str):
    """
    Full training loop for V_phi.

    Args:
        config:        Parsed prm_config.yaml as a nested dict.
        spider_dir:    Path to the Spider dataset root (for future use).
        processed_dir: Path to data/processed/ (output of build_corruption_dataset.py).
    """
    # Add src to path so imports resolve when called from scripts/
    src_dir = os.path.join(os.path.dirname(__file__), '..')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from data.dataset import PRMDataset
    from models.prm import ClausePRM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    mcfg = config['model']
    tcfg = config['training']
    ecfg = config['evaluation']
    pcfg = config['paths']

    os.makedirs(pcfg['output_dir'], exist_ok=True)
    log_dir = os.path.dirname(pcfg['log_file'])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg['name'], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ────────────────────────────────────────────────────────────
    print("Building dataset ...")
    full_ds = PRMDataset(
        processed_dir=processed_dir,
        tokenizer=tokenizer,
        max_length=tcfg['max_length'],
        schema_dropout_prob=tcfg['schema_dropout_prob'],
        training=True,
    )
    n_dev   = max(1, int(0.1 * len(full_ds)))
    n_train = len(full_ds) - n_dev
    train_ds, dev_ds = random_split(full_ds, [n_train, n_dev])
    # Disable dropout on dev split
    dev_ds.dataset.schema_dropout_prob = 0.0

    train_loader = DataLoader(
        train_ds, batch_size=tcfg['batch_size'],
        shuffle=True, num_workers=2, pin_memory=(device.type == 'cuda'),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=tcfg['batch_size'] * 2,
        shuffle=False, num_workers=2, pin_memory=(device.type == 'cuda'),
    )
    print(f"Train items: {len(train_ds):,}  Dev items: {len(dev_ds):,}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("Loading model ...")
    use_4bit = mcfg.get('quantization', '4bit') == '4bit'
    model = ClausePRM(
        model_name=mcfg['name'],
        lora_rank=mcfg['lora_rank'],
        lora_alpha=mcfg['lora_alpha'],
        lora_dropout=mcfg.get('lora_dropout', 0.05),
        target_modules=mcfg.get('target_modules', ['q_proj', 'v_proj']),
        use_4bit=use_4bit,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg['learning_rate'],
        weight_decay=tcfg['weight_decay'],
    )
    total_steps = max(1,
        len(train_loader) * tcfg['num_epochs'] // tcfg['grad_accumulation_steps']
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=tcfg['warmup_steps'],
        num_training_steps=total_steps,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    bce_loss_fn = nn.BCELoss(reduction='mean')
    lam         = tcfg.get('schema_grounding_lambda', 0.1)

    global_step = 0
    best_gap    = -float('inf')
    accum_loss  = 0.0
    log_entries: list[dict] = []

    for epoch in range(tcfg['num_epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tcfg['num_epochs']}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                scores = model(ids, mask)
                bce    = bce_loss_fn(scores, labels)

                # Schema-grounding auxiliary loss:
                # Decode batch back to text and compute per-example grounding flags.
                # Penalise high PRM scores for prefixes with out-of-schema tokens.
                grounding_loss = torch.tensor(0.0, device=device)
                if lam > 0 and device.type == 'cuda':
                    texts    = tokenizer.batch_decode(ids, skip_special_tokens=True)
                    sg_flags = torch.tensor(
                        [_schema_grounding_flag(t) for t in texts],
                        dtype=torch.float, device=device,
                    )
                    grounding_loss = lam * (scores * sg_flags).mean()

                loss = (bce + grounding_loss) / tcfg['grad_accumulation_steps']

            loss.backward()
            accum_loss += loss.item() * tcfg['grad_accumulation_steps']

            if (step + 1) % tcfg['grad_accumulation_steps'] == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    tcfg['max_grad_norm'],
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % ecfg['log_every'] == 0:
                    avg_loss   = accum_loss / ecfg['log_every']
                    accum_loss = 0.0
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", step=global_step)

                if global_step % ecfg['eval_every'] == 0:
                    dev_bce, gap = _compute_score_gap(model, dev_loader, device)
                    print(f"\n[Step {global_step}] dev_bce={dev_bce:.4f}  score_gap={gap:.4f}")
                    entry = {'step': global_step, 'epoch': epoch + 1,
                             'dev_bce': dev_bce, 'score_gap': gap}
                    log_entries.append(entry)
                    with open(pcfg['log_file'], 'a') as f:
                        f.write(json.dumps(entry) + '\n')

                    if gap > best_gap:
                        best_gap  = gap
                        ckpt_path = os.path.join(pcfg['output_dir'], 'best_checkpoint')
                        model.backbone.save_pretrained(ckpt_path)
                        tokenizer.save_pretrained(ckpt_path)
                        print(f"  Saved best checkpoint → {ckpt_path}  (gap={gap:.4f})")

                    model.train()

        # End of epoch checkpoint
        ckpt_path = os.path.join(pcfg['output_dir'], f'epoch_{epoch + 1}')
        model.backbone.save_pretrained(ckpt_path)
        print(f"Epoch {epoch + 1} checkpoint saved → {ckpt_path}")

    print(f"\nTraining complete. Best score_gap: {best_gap:.4f}")
    return log_entries
