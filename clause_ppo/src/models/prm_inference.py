"""
PRM inference — load a trained ClausePRM checkpoint and score clause prefixes.

Usage (Python API):
    from models.prm_inference import PRMScorer
    scorer = PRMScorer('/path/to/checkpoint', base_model='/path/to/qwen')
    score = scorer.score(question, schema, prefix_sql)   # float in (0, 1)
    scores = scorer.score_batch([(q, s, p), ...])        # list of floats

Usage (CLI):
    python scripts/score_clause.py \
        --checkpoint results/prm_checkpoints/best_checkpoint \
        --base_model /home/henrylin0822/models/qwen \
        --question "How many singers are there?" \
        --schema "concert_singer: singer(singer_id, name), concert(concert_id, name)" \
        --prefix "SELECT count(*) FROM singer"
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig


class PRMScorer:
    """Loads a trained PRM checkpoint and scores clause prefixes."""

    INPUT_TEMPLATE = "[QUESTION] {question} [SCHEMA] {schema} [PREFIX] {prefix}"

    def __init__(
        self,
        checkpoint_dir: str,
        base_model: str = None,
        max_length: int = 512,
        device: str = None,
    ):
        """
        Args:
            checkpoint_dir: Path to a saved checkpoint directory (contains
                            adapter_config.json, score_head.pt, tokenizer files).
            base_model:     Path or HF name of the base model. If None, reads
                            base_model_name_or_path from the adapter config.
            max_length:     Max token length for inputs.
            device:         'cuda', 'cpu', or None (auto-detect).
        """
        self.max_length = max_length
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        print(f"Loading PRM from {checkpoint_dir} on {self.device} ...", flush=True)

        # ── Resolve base model path ───────────────────────────────────────
        if base_model is None:
            peft_cfg = PeftConfig.from_pretrained(checkpoint_dir)
            base_model = peft_cfg.base_model_name_or_path
        print(f"  Base model: {base_model}", flush=True)

        # ── Tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Backbone (base + LoRA adapter) ────────────────────────────────
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Loading base model ...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map='auto',
            dtype=torch.bfloat16,
        )
        print("  Applying LoRA adapter ...", flush=True)
        self.backbone = PeftModel.from_pretrained(base, checkpoint_dir)
        self.backbone.eval()

        # ── Score head ────────────────────────────────────────────────────
        hidden_size = self.backbone.config.hidden_size
        score_head_path = os.path.join(checkpoint_dir, 'score_head.pt')
        self.score_head = nn.Linear(hidden_size, 1)
        if os.path.exists(score_head_path):
            self.score_head.load_state_dict(
                torch.load(score_head_path, map_location='cpu')
            )
            print("  Score head loaded.", flush=True)
        else:
            print("  WARNING: score_head.pt not found — using random weights.", flush=True)

        backbone_device = next(self.backbone.parameters()).device
        self.score_head = self.score_head.to(backbone_device)
        self.score_head.eval()

        print("PRM ready.", flush=True)

    @torch.no_grad()
    def score(self, question: str, schema: str, prefix_sql: str) -> float:
        """Score a single clause prefix. Returns probability in (0, 1)."""
        return self.score_batch([(question, schema, prefix_sql)])[0]

    @torch.no_grad()
    def score_batch(self, examples: list[tuple[str, str, str]]) -> list[float]:
        """Score a batch of (question, schema, prefix_sql) triples."""
        texts = [
            self.INPUT_TEMPLATE.format(question=q, schema=s, prefix=p)
            for q, s, p in examples
        ]
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        ids  = enc['input_ids'].to(next(self.backbone.parameters()).device)
        mask = enc['attention_mask'].to(ids.device)

        outputs = self.backbone(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]
        seq_lengths = mask.sum(dim=1) - 1
        batch_idx   = torch.arange(last_hidden.size(0), device=last_hidden.device)
        last_token  = last_hidden[batch_idx, seq_lengths]

        logits = self.score_head(last_token).squeeze(-1)
        probs  = torch.sigmoid(logits).float().cpu().tolist()
        return probs if isinstance(probs, list) else [probs]
