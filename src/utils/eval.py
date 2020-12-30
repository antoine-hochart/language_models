import numpy as np
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader

######################################################################
# n-gram

def get_perplexity(model, text):
    cross_entropy = 0
    n_tokens = 0
    try:
        for sentence in text:
            n_tokens += len(sentence) - 1
            cross_entropy += sum(-np.log(model.proba(token, sentence[:i+1]))
                                 for i, token in enumerate(sentence[1:]))
        perplexity = np.exp(cross_entropy / n_tokens)
    except (TypeError, ZeroDivisionError):
        perplexity = None
    return perplexity


def plot_perplexity(x, scores, labels, title='', figsize=(7,5)):
    plt.figure(figsize=figsize)
    for score, label in zip(scores, labels):
        plt.plot(x, score, label=label)
    plt.xlabel('discount weight')
    plt.ylabel('perplexity')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()

######################################################################
# neural network

def perplexity_loss(model, textset, batch_size, device):
    dataloader = DataLoader(textset, batch_size)
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    model.eval()
    cross_entropy = 0
    for inputs, labels in tqdm(dataloader, ascii=True):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        cross_entropy += loss.item() * inputs.size(0)
    cross_entropy = cross_entropy / len(textset)
    return np.exp(cross_entropy)
