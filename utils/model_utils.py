from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Positional encoding (sine/cosine)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        # x_q, x_kv: (batch, seq, d_model)
        B, Tq, _ = x_q.shape
        B, Tk, _ = x_kv.shape

        q = self.q_proj(x_q).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tq, D)
        k = self.k_proj(x_kv).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Tk, D)
        v = self.v_proj(x_kv).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, Tq, Tk)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # assume mask has -inf where masked

        if key_padding_mask is not None:
            # key_padding_mask: (B, Tk) True for PAD positions
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,Tk)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, Tq, D)
        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.o_proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor] = None):
        # Self-attention
        sa = self.self_attn(x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(sa))
        # FF
        ff = self.ff(x)
        x = self.norm2(x + self.dropout(ff))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None):
        # Masked self-attention (causal)
        sa = self.self_attn(x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(sa))
        # Cross-attention
        ca = self.cross_attn(x, enc_out, key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(ca))
        # FF
        ff = self.ff(x)
        x = self.norm3(x + self.dropout(ff))
        return x

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 d_ff: int = 2048, dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_decoder_layers)])

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings and output projection (optional but common)
        self.lm_head.weight = self.tgt_embed.weight

    def generate_square_subsequent_mask(self, sz: int, device: torch.device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float("-inf")).masked_fill(~mask, 0.0)
        return mask  # (T, T) additive mask

    def forward(self, src_ids: torch.Tensor, tgt_in_ids: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None):
        # Shapes:
        # src_ids: (B, S) ; tgt_in_ids: (B, T_in)
        B, S = src_ids.shape
        B2, T = tgt_in_ids.shape
        assert B == B2

        src = self.src_embed(src_ids) * math.sqrt(self.d_model)
        tgt = self.tgt_embed(tgt_in_ids) * math.sqrt(self.d_model)
        src = self.pos_enc(src)
        tgt = self.pos_enc(tgt)

        # Encoder
        enc = src
        for layer in self.encoder:
            enc = layer(enc, src_key_padding_mask=src_key_padding_mask)

        # Decoder
        tgt_mask = self.generate_square_subsequent_mask(T, src_ids.device)  # (T, T)
        # Convert tgt_mask to (1, T, T) to broadcast over batch and heads
        tgt_mask = tgt_mask.unsqueeze(0)

        dec = tgt
        for layer in self.decoder:
            dec = layer(dec, enc, tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask)

        logits = self.lm_head(dec)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids: torch.Tensor, max_len: int, bos_id: int, eos_id: int) -> torch.Tensor:
        device = src_ids.device
        B = src_ids.size(0)
        src_key_padding_mask = (src_ids == self.pad_id)
        enc = self.src_embed(src_ids) * math.sqrt(self.d_model)
        enc = self.pos_enc(enc)
        for layer in self.encoder:
            enc = layer(enc, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt = self.tgt_embed(ys) * math.sqrt(self.d_model)
            tgt = self.pos_enc(tgt)
            T = ys.size(1)
            tgt_mask = self.generate_square_subsequent_mask(T, device).unsqueeze(0)
            dec = tgt
            for layer in self.decoder:
                dec = layer(dec, enc, tgt_mask=tgt_mask,
                            tgt_key_padding_mask=(ys == self.pad_id),
                            memory_key_padding_mask=src_key_padding_mask)
            logits = self.lm_head(dec)[:, -1, :]  # last token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return ys

    @torch.no_grad()
    def beam_search(self, src_ids: torch.Tensor, max_len: int, bos_id: int, eos_id: int, beam_size: int = 4, length_penalty: float = 0.6):
        # Simple batch-agnostic beam search (batch size 1 recommended)
        assert src_ids.size(0) == 1, "beam_search currently supports batch size 1"
        device = src_ids.device
        src_key_padding_mask = (src_ids == self.pad_id)

        enc = self.src_embed(src_ids) * math.sqrt(self.d_model)
        enc = self.pos_enc(enc)
        for layer in self.encoder:
            enc = layer(enc, src_key_padding_mask=src_key_padding_mask)

        beams = [(torch.tensor([[bos_id]], device=device, dtype=torch.long), 0.0)]  # (tokens, logprob)
        finished = []

        for _ in range(max_len - 1):
            new_beams = []
            for tokens, score in beams:
                if tokens[0, -1].item() == eos_id:
                    finished.append((tokens, score))
                    continue
                tgt = self.tgt_embed(tokens) * math.sqrt(self.d_model)
                tgt = self.pos_enc(tgt)
                T = tokens.size(1)
                tgt_mask = self.generate_square_subsequent_mask(T, device).unsqueeze(0)
                dec = tgt
                for layer in self.decoder:
                    dec = layer(dec, enc, tgt_mask=tgt_mask,
                                tgt_key_padding_mask=(tokens == self.pad_id),
                                memory_key_padding_mask=src_key_padding_mask)
                logits = self.lm_head(dec)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)
                for k in range(beam_size):
                    new_toks = torch.cat([tokens, topk_ids[:, k:k+1]], dim=1)
                    new_score = score + topk_log_probs[0, k].item()
                    new_beams.append((new_toks, new_score))
            # keep top beams
            new_beams.sort(key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)
            beams = new_beams[:beam_size]

        finished.extend(beams)
        finished.sort(key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)
        return finished[0][0]