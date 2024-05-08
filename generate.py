import torch
from torch.nn import functional as F
import os

from utils import get_tokenizer
from model import Model, ModelArgs

model_path = os.getcwd() + "/model/snapshot.pt"

model_args = ModelArgs()
model = Model(model_args)

model.load_state_dict(torch.load(model_path))
model.eval()

tokenizer = get_tokenizer()
model_path = os.getcwd() + "/model/snapshot.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

text = "<user>merhabalar<bot>merhaba bu gün size nasıl yardımcı olabilirim<user>Fırat üniversitesinde kaç adet bölüm vardır?<bot>"

def generate_text(model, text: str, stop_token, max_token: int = 100, temprature: float = 1.0):
    model.eval()
    idx = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_token):
            logits, _ = model(idx)
            logits = logits[:, -1, :] / temprature
            props = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(props, num_samples=1)
            print(idx_next)
            if idx_next in stop_token:
                break
            idx = torch.cat((idx, idx_next), dim=1)
            
    return tokenizer.decode(idx[0].tolist())

a = generate_text(model, text, stop_token=[32001])
print(a)
