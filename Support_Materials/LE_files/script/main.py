import os
import sys
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

basedir = os.path.abspath(os.path.dirname(__file__))

class MyDataset(Dataset):
    def __init__(self, mode, tokenizer):
        self.mode = mode
        self.data = []
        for file in os.listdir(os.path.join(basedir, 'task3', 'ELE', self.mode)):
            with open(os.path.join(basedir, 'task3', 'ELE', self.mode, file)) as f:
                json_data = json.load(f)
                raw = list(map(lambda s:tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)), json_data['article'].split('_')))
                question_cnt = len(json_data['answers'])
                prev = raw[0]
                for i in range(question_cnt):
                    ans = json_data["options"][i][ord(json_data["answers"][i]) - ord('A')]
                    sentence = prev + [tokenizer.mask_token_id] + raw[i + 1]
                    label = prev + [tokenizer.convert_tokens_to_ids(ans)] + raw[i + 1]
                    self.data.append((torch.tensor(sentence), torch.tensor(label)))
                    prev = label
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

def create_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    label_tensors = [s[1] for s in samples]
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    label_tensors = pad_sequence(label_tensors, batch_first=True)
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    return tokens_tensors, masks_tensors, label_tensors


tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
model = torch.hub.load('huggingface/pytorch-transformers', 'modelWithLMHead', 'bert-base-uncased')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)

model.to(device)

def train():
    dataset = MyDataset('train', tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=create_batch)
    for epoch in range(10):
        for data in dataloader:
            token_tensor, masks_tensor, label_tensor = [t.to(device) for t in data]
            optimizer.zero_grad()
            output = model(token_tensor, labels=label_tensor, attention_mask=masks_tensor)
            loss = output.loss
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch}, loss {loss.item()}')

def validate(file_name):
    with open(file_name) as f:
        json_data = json.load(f)
    raw = json_data['article'].split('_')
    question_cnt = len(json_data['options'])
    prev = raw[0]
    correct = 0
    for i in range(question_cnt):
        sentence = prev[-255:] + tokenizer.mask_token + raw[i + 1][:255]
        tokens = tokenizer.encode(sentence)
        token_tensor = torch.tensor([tokens]).to(device)
        with torch.no_grad():
            pred = model(token_tensor)[0]
        masked_idx = tokens.index(tokenizer.mask_token_id)
        probs = pred[0, masked_idx, :].softmax(dim=0).tolist()
        max_prob, choice = 0.0, 0
        for j in range(4):
            id = tokenizer.convert_tokens_to_ids(json_data["options"][i][j])
            if max_prob < probs[id]:
                max_prob = probs[id]
                choice = j
        correct += choice == ord(json_data["answers"][i]) - ord('A')
        prev = prev + json_data["options"][i][choice] + raw[i + 1]
    return correct / question_cnt

def test(file_name):
    with open(file_name) as f:
        json_data = json.load(f)
    ans = []
    raw = json_data['article'].split('_')
    question_cnt = len(json_data['options'])
    prev = raw[0]
    correct = 0
    for i in range(question_cnt):
        sentence = prev[-255:] + tokenizer.mask_token + raw[i + 1][:255]
        tokens = tokenizer.encode(sentence)
        token_tensor = torch.tensor([tokens]).to(device)
        with torch.no_grad():
            pred = model(token_tensor)[0]
        masked_idx = tokens.index(tokenizer.mask_token_id)
        probs = pred[0, masked_idx, :].softmax(dim=0).tolist()
        max_prob, choice = 0.0, 0
        for j in range(4):
            id = tokenizer.convert_tokens_to_ids(json_data["options"][i][j])
            if max_prob < probs[id]:
                max_prob = probs[id]
                choice = j
        ans.append(chr(choice + ord('A')))
        prev = prev + json_data["options"][i][choice] + raw[i + 1]
    return ans

if __name__ == "__main__":
    test_dir = os.path.join(basedir, 'task3', 'ELE', 'test')
    ans = {}
    for file in os.listdir(test_dir):
        id = file.split('.')[0]
        choices = test(os.path.join(test_dir, file))
        print(id, choices)
        ans[id] = choices
    with open('answer.json', 'w') as f:
        json.dump(ans, f)
