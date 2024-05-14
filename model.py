import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelArgs:
    block_size: int = 1024
    vocab_size: int = 32002
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-4


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.n_embd % args.n_head == 0
        self.to_qkv = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.to_out = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(args.block_size, args.block_size))
                                .view(1, 1, args.block_size, args.block_size))
            
    def forward(self, x: torch.Tensor):
        B, T, C = x.size()

        q, k, v = self.to_qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, 
                k, 
                v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.to_out(y))
        return y
    
class FeedForward(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.to_fi = nn.Linear(args.n_embd, 4 * args.n_embd, bias=args.bias)
        self.to_out = nn.Linear(args.n_embd * 4, args.n_embd, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.to_fi(x)
        x = self.gelu(x)
        x = self.to_out(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.attention_norm = LayerNorm(args.n_embd, bias=args.bias)
        self.ffn_norm = LayerNorm(args.n_embd, bias=args.bias)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
    

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.block_size is not None

        self.args = args

        self.tok_embedding = nn.Embedding(args.vocab_size, args.n_embd)
        self.pos_embedding = nn.Embedding(args.block_size, args.n_embd)
        self.drop = nn.Dropout(args.dropout)
        self.ln_norm = LayerNorm(args.n_embd, bias=args.bias)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layer):
            self.layers.append(TransformerBlock(layer_id, args))

        self.output = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.args.block_size, f"Cannot forward sequence of length {T}, block size is only {self.args.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        idx = idx.to(torch.long)
        token_emb = self.tok_embedding(idx)
        positional_emb = self.pos_embedding(pos)
        x = self.drop(token_emb + positional_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_norm(x)

        if targets is not None:
            targets = targets.to(torch.long)
            logits = self.output(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.output(x)
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tokens, do_sample: bool = False, temprature: float = 1.0, top_k: Optional[int] = None):

        for _ in range(max_new_tokens):
            # sequence cropping
            idx_crop = idx if idx.size(1) <= self.args.block_size else idx[:, -self.args.block_size:]

            logits, _ = self(idx_crop)

            logits = logits[:, -1, :] / temprature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-Inf")

            props = F.softmax(logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(props, num_samples=1)
            else:
                _, idx_next = torch.topk(props, k=1, dim=-1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx