from dataclasses import dataclass
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast

from typing import Optional

from model import Transformer
from prepare_data import preprocess_dialogues, prepare_text_data
from utils import get_tokenizer
from lora import Lora

device = 'cuda' if torch.cuda.is_available() else 'cpu'

scaler = GradScaler()

batch_size = 2
block_size = 1024
max_iters = 200
eval_interval = 50
learning_rate = 4e-4
eval_iters = 50
dropout = 0.2
train_type = "pretrain"

model_path = os.path.join(os.getcwd(), "model", "snapshot.pt")
model_dir = os.path.join(os.getcwd(), "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

@dataclass
class ModelArgs:
    block_size: int = 1024
    vocab_size: int = 32002
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1e-4


torch.manual_seed(42)

tokenizer = get_tokenizer()

json_data_path = os.path.join(os.getcwd(), "data", "intents.json")
train_data, val_data = preprocess_dialogues(json_data_path)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

model_args = ModelArgs()
model = Transformer(model_args)
print("model loading...")
model.load_state_dict(torch.load(model_path,  map_location=device))
model.to(device)
print("lora weight adding...")
lora = Lora(model)
lora.freeze_non_lora_params()
lora.print_model_parameters()
lora.enable_disable_lora(enabled=True)
total_params, trainable_params = lora.count_parameters()
print(f"Toplam parametre sayısı: {total_params}")
print(f"Eğitilebilir parametre sayısı: {trainable_params}")
model = lora.model

if train_type == "pretrain":
    model_args = ModelArgs()
    model = Transformer(model_args)
    print("model loading...")
    model.load_state_dict(torch.load(model_path,  map_location=device))
    model.to(device)
    print("lora weight adding...")
    lora = Lora(model)
    lora.freeze_non_lora_params()
    lora.print_model_parameters()
    lora.enable_disable_lora(enabled=True)
    total_params, trainable_params = lora.count_parameters()
    print(f"Toplam parametre sayısı: {total_params}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params}")
    model = lora.model
else:
    model_args = ModelArgs()
    model = Transformer(model_args)
    #model.load_state_dict(torch.load(model_path,  map_location=device))
    model.to(device)
    lora = Lora(model)
    lora.freeze_non_lora_params()
    lora.print_model_parameters()
    lora.enable_disable_lora(enabled=True)
    total_params, trainable_params = lora.count_parameters()
    print(f"Toplam parametre sayısı: {total_params}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params}")
    model = lora.model
    model.load_state_dict(torch.load(model_path,  map_location=device))



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for iter in range(max_iters):

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    optimizer.zero_grad(set_to_none=True)
    with autocast(dtype=torch.float16):
        logits, loss = model(xb, yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


text = "Bilgisayar Mühendisliği"
context = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
print(tokenizer.decode(model.generate(context, max_new_tokens=400)[0].tolist()))

torch.save(model.state_dict(), model_path)
