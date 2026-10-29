"""PathoJev: direct probabilistic decisions for whole-slide and region-level pathology.

A frozen Qwen3.5 backbone with a LoRA adapter, a 24-slot typed decision head and a slide entry that maps a 32 x 32 grid
of pooled CONCH v1.5 features to slide tokens.

Whole slide: "[state] Whole-slide image (H&E): <slide tokens>\\n" is encoded once; every question is an isolated row
"[kind] ..\\n[question] ..\\n[options]\\n..\\n[decision]:" read out by the decision head at its last token.
ROI image: one row per image through the backbone's own vision tower.
"""
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file

from . import prefix as jev

LETTERS = "ABCDEFGHIJKLMNOP"
SLIDE_STATE = "[state] Whole-slide image (H&E): "
ROI_STATE = "[state] An H&E-stained histopathology image: <|vision_start|><|image_pad|><|vision_end|>\n"
SLOTS = {"noul": 0, "score": 2, "choice": 8}  # first slot of each output type (2 binary, 6 ordinal, 16 choice slots)


def row_text(q, image=False):
    """Question row in JEV's bare-v1 format. q: {"question", "options"} (choice) or {"question", "kind": "noul"}
    (true / false statement). ROI rows put [kind] before [state]."""
    kind = q.get("kind", "choice")
    lines = ["false", "true"] if kind == "noul" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(q["options"])]
    body = f"[question] {q['question']}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    return f"[kind] {kind}\n{ROI_STATE}{body}" if image else f"[kind] {kind}\n{body}"


class SlideEntry(nn.Module):
    def __init__(self, feat_dim, d, grid):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(feat_dim, d), nn.GELU(), nn.Linear(d, d))
        self.row = nn.Embedding(grid, d)
        self.col = nn.Embedding(grid, d)

    def forward(self, x, rc):
        return self.mlp(x) + self.row(rc[:, 0]) + self.col(rc[:, 1])


class PathoJev(nn.Module):
    def __init__(self, weights, config, base=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
        cfg = json.load(open(config))
        base = base or cfg["base_model"]
        self.proc = AutoProcessor.from_pretrained(base, max_pixels=cfg["max_pixels"])
        self.proc.tokenizer.padding_side = "right"
        self.tok = self.proc.tokenizer
        vl = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
        vl.config.text_config._attn_implementation = jev.IMPL  # language model only; the vision tower keeps SDPA
        vl.requires_grad_(False)
        targets = [n for n, _ in vl.named_modules() if n.split(".")[-1] in cfg["lora"]["target_modules"] and "visual" not in n]
        self.vl = get_peft_model(vl, LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], target_modules=targets))
        d = vl.config.text_config.hidden_size
        self.head = nn.Linear(d, 24)
        self.slide = SlideEntry(cfg["feat_dim"], d, cfg["grid"])
        sd = load_file(weights)
        missing = set(sd) - {n for n, _ in self.named_parameters()}
        assert not missing, sorted(missing)[:3]
        self.load_state_dict(sd, strict=False)
        self.eval()

    @classmethod
    def from_pretrained(cls, size="9b", repo="Mikezcy/PathoJev", base=None, device="cuda"):
        """size: "9b" or "27b"; repo: Hugging Face repo id or a local copy of it; base: optional local copy of the
        backbone (Qwen/Qwen3.5-9B for 9b, autotrust/JEV-27B-VL for 27b)."""
        if os.path.isdir(repo):
            w, c = os.path.join(repo, size, "pathojev.safetensors"), os.path.join(repo, size, "config.json")
        else:
            from huggingface_hub import hf_hub_download
            w, c = (hf_hub_download(repo, f"{size}/{f}") for f in ("pathojev.safetensors", "config.json"))
        return cls(w, c, base).to(device)

    def _ids(self, t):
        return self.tok(t, add_special_tokens=False)["input_ids"]

    @property
    def lm(self):
        return self.vl.get_base_model().model.language_model

    @property
    def device(self):
        return self.slide.row.weight.device

    def _probs(self, h, qs):
        z = self.head(h.float())
        out = []
        for zi, q in zip(z, qs):
            kind = q.get("kind", "choice")
            n = 2 if kind == "noul" else len(q["options"])
            out.append(torch.softmax(zi[SLOTS[kind]:SLOTS[kind] + n], -1))  # temperature 1, no calibration
        return out

    @staticmethod
    def _result(q, p):
        labels = ["false", "true"] if q.get("kind", "choice") == "noul" else q["options"]
        p = p.tolist()
        return {"answer": labels[max(range(len(p)), key=p.__getitem__)], "probs": dict(zip(labels, p))}

    @torch.no_grad()
    def answer_slide(self, grid, questions, chunk=64):
        """grid: {"x": (n, 768) pooled CONCH v1.5 features, "rc": (n, 2) grid row/col} (see make_grid.py);
        questions: list of question dicts. The slide prefix is encoded once; questions are read out as isolated rows."""
        emb = self.lm.embed_tokens
        x, rc = grid["x"].to(self.device).float(), grid["rc"].to(self.device).long()
        z = self.slide(x, rc).to(torch.bfloat16)
        pre = torch.cat([emb(torch.tensor(self._ids(SLIDE_STATE), device=self.device)), z,
                         emb(torch.tensor(self._ids("\n"), device=self.device))])[None]
        kv, P = jev.prefix_state(self.lm(inputs_embeds=pre, use_cache=True).past_key_values), pre.shape[1]
        res = []
        for s in range(0, len(questions), chunk):
            qs = questions[s:s + chunk]
            rows = [self._ids(row_text(q)) for q in qs]
            L = max(map(len, rows))
            ids = torch.tensor([r + [self.tok.pad_token_id or 0] * (L - len(r)) for r in rows], device=self.device)
            pos = (P + torch.arange(L, device=self.device))[None].expand(len(rows), -1)
            h = self.lm(inputs_embeds=emb(ids), position_ids=pos, jev_prefix=kv, use_cache=False).last_hidden_state
            last = torch.tensor([len(r) - 1 for r in rows], device=self.device)
            res += [self._result(q, p) for q, p in zip(qs, self._probs(h[torch.arange(len(rows)), last], qs))]
        return res

    @torch.no_grad()
    def answer_roi(self, images, questions):
        """images: list of PIL images (one per question); questions: list of question dicts."""
        enc = self.proc(text=[row_text(q, image=True) for q in questions], images=images, padding=True,
                        return_tensors="pt", add_special_tokens=False).to(self.device)
        h = self.vl.get_base_model().model(**enc, use_cache=False).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1
        return [self._result(q, p) for q, p in zip(questions, self._probs(h[torch.arange(h.shape[0]), last], questions))]
