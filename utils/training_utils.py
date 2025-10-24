from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

@dataclass
class TrainConfig:
    d_model: int = 384
    num_heads: int = 6
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_epochs: int = 20
    patience: int = 3  # early stopping patience
    label_smoothing: float = 0.1
    max_src_len: int = 128
    max_tgt_len: int = 128
    max_tokens_per_microbatch: int = 2048  # adaptive, will shrink on OOM
    grad_accum_steps: int = 8
    clip_grad_norm: float = 1.0
    beam_size: int = 4
    length_penalty: float = 0.6
    seed: int = 42

class TokenBatchDataset(Dataset):
    def __init__(self, src_ids, tgt_ids):
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]

def make_pad_mask(lengths, max_len, device):
    # returns mask True at PAD positions
    B = len(lengths)
    idxs = torch.arange(0, max_len, device=device).unsqueeze(0).expand(B, -1)
    lens = torch.tensor(lengths, device=device).unsqueeze(1)
    mask = idxs >= lens
    return mask  # (B, max_len)

def collate_dynamic(samples, pad_id: int, max_src_len: int, max_tgt_len: int):
    # samples: list of (src_ids, tgt_ids) as python lists
    batch_src = []
    batch_tgt_in = []
    batch_tgt_out = []
    src_lens = []
    tgt_lens = []
    for src, tgt in samples:
        # truncate
        src = src[:max_src_len]
        tgt = tgt[:max_tgt_len-1]  # leave room for EOS in _out
        # Prepare teacher-forcing: input (BOS + y[:-1]), output (y)
        # Here we assume the data already has BOS at start and EOS at end
        tgt_in = tgt[:-1]
        tgt_out = tgt[1:]

        batch_src.append(src)
        batch_tgt_in.append(tgt_in)
        batch_tgt_out.append(tgt_out)
        src_lens.append(len(src))
        tgt_lens.append(len(tgt_in))

    max_s = max(src_lens)
    max_t = max(tgt_lens)
    B = len(samples)

    src_ids = torch.full((B, max_s), pad_id, dtype=torch.long)
    tgt_in_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    tgt_out_ids = torch.full((B, max_t), pad_id, dtype=torch.long)

    for i in range(B):
        s = batch_src[i]
        t_in = batch_tgt_in[i]
        t_out = batch_tgt_out[i]
        src_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_in_ids[i, :len(t_in)] = torch.tensor(t_in, dtype=torch.long)
        tgt_out_ids[i, :len(t_out)] = torch.tensor(t_out, dtype=torch.long)

    return src_ids, tgt_in_ids, tgt_out_ids, src_lens, tgt_lens

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing: float, vocab_size: int, ignore_index: int):
        super().__init__()
        self.ignore_index = ignore_index
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred: (B, T, V) logits
        # target: (B, T)
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)
        mask = target != self.ignore_index
        target = target[mask]
        pred = pred[mask]
        true_dist = torch.full_like(pred, self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        log_probs = torch.log_softmax(pred, dim=-1)
        loss = (-true_dist * log_probs).sum(dim=1).mean()
        return loss

class WarmupInverseSqrtScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = min(step ** (-0.5), step * (self.warmup_steps ** (-1.5)))
        return [base_lr * scale for base_lr in self.base_lrs]

def estimate_tokens_in_batch(samples) -> int:
    # Estimate tokens in collate (sum of src + tgt_in)
    total = 0
    for src, tgt in samples:
        total += len(src) + max(1, len(tgt) - 1)
    return max(1, total)

def build_adaptive_dataloader(dataset: TokenBatchDataset, pad_id: int, cfg: TrainConfig,
                              shuffle: bool = True):
    # Create buckets by length for efficiency
    indices = list(range(len(dataset)))
    if shuffle:
        import random; random.Random(cfg.seed).shuffle(indices)

    # Group into micro-batches with a token budget
    microbatches = []
    buf = []
    current_tokens = 0
    for idx in indices:
        buf.append(dataset[idx])
        current_tokens += len(dataset[idx][0]) + len(dataset[idx][1])
        if current_tokens >= cfg.max_tokens_per_microbatch:
            microbatches.append(buf)
            buf = []
            current_tokens = 0
    if buf:
        microbatches.append(buf)

    def collate_fn(samples):
        return collate_dynamic(samples, pad_id, cfg.max_src_len, cfg.max_tgt_len)

    # Return an iterator over micro-batches (lists), not a standard DataLoader
    return microbatches, collate_fn

def train_one_epoch(model, optimizer, scheduler, scaler, train_batches, collate_fn, device, pad_id, cfg: TrainConfig):
    model.train()
    loss_fn = LabelSmoothingLoss(cfg.label_smoothing, model.vocab_size, ignore_index=pad_id)
    total_loss = 0.0
    step = 0
    accum = 0

    optimizer.zero_grad(set_to_none=True)

    for micro in tqdm(train_batches, desc="Train micro-batches"):
        # Adaptive microbatch retry loop for OOM safety
        success = False
        cur = micro
        while not success and len(cur) > 0:
            try:
                src_ids, tgt_in_ids, tgt_out_ids, src_lens, tgt_lens = collate_fn(cur)
                src_ids = src_ids.to(device, non_blocking=True)
                tgt_in_ids = tgt_in_ids.to(device, non_blocking=True)
                tgt_out_ids = tgt_out_ids.to(device, non_blocking=True)

                with autocast():
                    logits = model(src_ids, tgt_in_ids,
                                   src_key_padding_mask=(src_ids == pad_id),
                                   tgt_key_padding_mask=(tgt_in_ids == pad_id))
                    loss = loss_fn(logits, tgt_out_ids) / cfg.grad_accum_steps

                scaler.scale(loss).backward()
                accum += 1

                if accum % cfg.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1

                total_loss += loss.item() * cfg.grad_accum_steps
                success = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    # Split the current microbatch in half and retry
                    if len(cur) >= 2:
                        mid = len(cur) // 2
                        cur = cur[:mid]
                    else:
                        # Drop the sample if it's still OOM (very long sequence)
                        cur = []
                else:
                    raise e
    return total_loss / max(1, len(train_batches))

@torch.no_grad()
def evaluate_bleu(model, valid_batches, collate_fn, device, pad_id, spm, bos_id, eos_id, cfg: TrainConfig):
    from sacrebleu import corpus_bleu
    model.eval()
    refs = []
    hyps = []
    for micro in tqdm(valid_batches, desc="Validate micro-batches"):
        src_ids, tgt_in_ids, tgt_out_ids, src_lens, tgt_lens = collate_fn(micro)
        src_ids = src_ids.to(device, non_blocking=True)
        # Greedy for speed; beam in final eval
        ys = model.greedy_decode(src_ids, max_len=cfg.max_tgt_len, bos_id=bos_id, eos_id=eos_id)
        ys = ys.cpu().tolist()
        for i in range(len(ys)):
            hyp = spm.decode(ys[i])
            ref = spm.decode(tgt_in_ids[i].cpu().tolist())  # approximate; better to load raw text
            hyps.append(hyp)
            refs.append([ref])
    score = corpus_bleu(hyps, list(zip(*refs))).score
    return score