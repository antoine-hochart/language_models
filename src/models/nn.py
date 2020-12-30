import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, seq_len):
        super(RNNLM, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size+1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        x  = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = h[-1].squeeze()
        out = self.fc(h)
        return out
    
    def prepare_input(self, history):
        X = torch.LongTensor(history)
        X = X[-self.seq_len:] # trimming
        if len(X) < self.seq_len: # padding
            zeros = torch.zeros(self.seq_len-len(X), dtype=torch.long)
            X = torch.cat((zeros, X))
        return X

    def proba(self, token, history):
        X = self.prepare_input(history)
        with torch.no_grad():
            y = self.forward(X)
            probas = F.softmax(y, dim=0)
        idx = token - 1 # indices of token are shifted by 1
        return probas[idx].item()

    def generate_token(self, history):
        X = self.prepare_input(history)
        with torch.no_grad():
            y = self.forward(X)
            probas = F.softmax(y, dim=0)
            probas = probas.detach().cpu().numpy()
        token = np.random.choice(range(self.vocab_size), p=probas)
        return token + 1

    def generate_text(self, seed, eos, n_sent=1):
        np.random.seed()
        text = []
        for _ in range(n_sent):
            sentence = seed.copy()
            while sentence[-1] not in eos:
                token = self.generate_token(sentence)
                sentence.append(token)
            text.append(sentence)
        return text