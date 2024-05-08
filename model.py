from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from typing import Optional

@dataclass
class ModelArgs:
    block_size: int = 1024
    vocab_size: int = 32002
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-4

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embedding = nn.Embedding(args.vocab_size, args.n_embd)
        self.pos_embedding = nn.Embedding(args.block_size, args.n_embd)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.n_embd, nhead=args.n_head)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=args.n_layer)
        self.output = nn.Linear(args.n_embd, args.vocab_size, bias=args.bias)
        self.drop = nn.Dropout(args.dropout)
        self.ln_norm = nn.LayerNorm(args.n_embd, bias=args.bias)

    def forward(self, idx: torch.Tensor, targets=None):
        device = idx.device
        B, T = idx.size()
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        idx = idx.to(torch.long)
        token_emb = self.tok_embedding(idx)
        positional_emb = self.pos_embedding(pos)
        x = self.drop(token_emb + positional_emb)

        # Rastgele bir 'memory' tensoru oluşturun
        memory = torch.rand(B, T, self.args.n_embd, device=device)

        x = self.transformer_decoder(tgt=x, memory=memory)

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

