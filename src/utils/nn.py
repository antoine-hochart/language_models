import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from time import time
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

######################################################################

def encode_text(text, vocab):
    # encoder dictionary (leave 0 index for padding)
    encoder = dict((word, i+1) for i, word in enumerate(vocab))
    if isinstance(text, str):
        res = encoder[text]
    elif isinstance(text, list) and isinstance(text[0], str):
        res = [encoder[word] for word in text]
    else:
        res = [[encoder[word] for word in sentence] for sentence in text]
    return res


def decode_text(text, vocab):
    return [[vocab[i-1] for i in sentence] for sentence in text]


class Textset(Dataset):
    def __init__(self, text, vocab, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.text = encode_text(text, vocab)
        self.indices = np.cumsum(
            [0] + [len(text[i-1]) - 1 for i in range(1, len(text))]
            ) # index of first item in each sentence
    
    def __len__(self):
        return sum(len(sentence)-1 for sentence in self.text)
    
    def __getitem__(self, idx):
        # get index of sentence
        k = np.where(idx - self.indices >= 0)[0][-1]
        # get index of token in sentence
        l = idx - self.indices[k] + 1
        # input
        X = torch.LongTensor(self.text[k][:l])
        X = X[-self.seq_len:] # trimming
        if len(X) < self.seq_len: # padding
            zeros = torch.zeros(self.seq_len-len(X), dtype=torch.long)
            X = torch.cat((zeros, X))
        # label (in encoded text, 0 is for padding -> shift by 1)
        y = torch.tensor(self.text[k][l] - 1)
        return X, y


class TrainValLoader():
    def __init__(self, train_text, val_text, vocab, seq_len,
                 train_batch_size=1, val_batch_size=1):
        train_set = Textset(train_text, vocab, seq_len)
        val_set = Textset(val_text, vocab, seq_len)
        self.train = DataLoader(train_set, train_batch_size,
                                shuffle=True, pin_memory=True)
        self.val = DataLoader(val_set, val_batch_size)

######################################################################

def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model):
    fpath = os.path.join(os.path.dirname(__file__), '..', '..',
                         'data', 'models', 'rnn.pt')
    fpath = os.path.abspath(fpath)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    torch.save(model.state_dict(), fpath)


def train(model, dataloader, optimizer, num_epochs, device, print_freq=1):
    if print_freq > 0:
        print("PyTorch {} + {}".format(torch.__version__, device))
        print("No. model parameters: {}".format(get_num_params(model)))
        print()
    
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_history = []
    val_loss_history = []
    best_loss = np.Inf

    model.to(device)

    since = time()

    for epoch in range(1, num_epochs+1):
        if print_freq > 0 and epoch % print_freq == 0:
            epoch_counter = '{}/{}'.format(epoch, num_epochs)
            print('Epoch {:>5}'.format(epoch_counter))
        # training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in tqdm(dataloader.train, ascii=True,
                                   disable=(epoch%print_freq != 0)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss = epoch_train_loss / len(dataloader.train.dataset)
        train_loss_history.append(epoch_train_loss)

        # validation phase
        model.eval()
        epoch_val_loss = 0.0
        for inputs, labels in tqdm(dataloader.val, ascii=True,
                                   disable=(epoch%print_freq != 0)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            epoch_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = epoch_val_loss / len(dataloader.val.dataset)
        val_loss_history.append(epoch_val_loss)

        # record best model
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            save_checkpoint(model)

        # monitoring
        if print_freq > 0 and epoch % print_freq == 0:
            print('train loss: {:09.3e}    val loss: {:09.3e}'.format(
                epoch_train_loss, epoch_val_loss))
        
    time_elapsed = time() - since
    if print_freq > 0:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed//60, time_elapsed%60))
        print('Best validation loss: {:.3e}'.format(best_loss))

    return train_loss_history, val_loss_history


def plot_loss(train_loss, val_loss, figsize=(7,5), logscale=True):
    fig, ax = plt.subplots(figsize=figsize)
    epochs = range(1, len(train_loss)+1)

    ax.plot(epochs, train_loss, label='train')
    ax.plot(epochs, val_loss, label='val')

    xlabel = 'Epoch'
    ylabel = 'Cross-entropy'
    if logscale:
        ax.set_yscale('log')
        ylabel += ' (log scale)'
    ax.set_xlabel(xlabel, {'fontsize': 12})
    ax.set_ylabel(ylabel, {'fontsize': 12}, rotation=90)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()

