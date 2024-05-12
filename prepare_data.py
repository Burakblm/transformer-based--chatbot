from utils import get_tokenizer
import json
from typing import List
import torch


tokenizer = get_tokenizer()
print(tokenizer.vocab_size)

def preprocess_dialogues(data_path: str, tokenizer) -> List[int]:

    with open(data_path, "r") as f:
        data = json.load(f)
    
    dialogue_array = []

    for dialogue in data:
        dialogue_text = ""
        for data in dialogue["dialog"]:
            if data["sender"] == "user":
                formatted_sentence = f"<user>{data['text']}\n"
            elif data["sender"] == "bot":
                formatted_sentence = f"<bot>{data['text']}\n"
            else:
                raise ValueError(f"Geçersiz gönderici: {data['sender']}")
            
            dialogue_text = dialogue_text + formatted_sentence
        dialogue_text = dialogue_text + tokenizer.eos_token
        dialogue_array.extend(tokenizer.encode(dialogue_text))

    return dialogue_array


def prepare_text_data(path: str = "output.txt", split_rate: float = 0.9):
    with open(path, "r") as f:
        data = f.read()
    
    data = tokenizer.encode(data)
    data = torch.tensor(data, dtype=torch.long)
    data_size = len(data)
    train_data = data[:int(data_size * split_rate)]
    print(f"Train data size: {len(train_data)}")
    val_data = data[int(data_size * split_rate):]
    print(f"Validation data size: {len(val_data)}")

    return train_data, val_data