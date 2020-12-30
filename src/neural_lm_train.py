import numpy as np
import torch

from time import time
from tqdm import tqdm

from utils.data import get_corpus, preprocess_text
from utils.nn import TrainValLoader, train, plot_loss
from models.nn import RNNLM

######################################################################
# parameters

MIN_COUNT = 5

EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
SEQ_LEN = 30
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 1024
NUM_EPOCHS = 20

######################################################################
# load and preprocess text

print("Loading and preprocessing text...")
t0 = time()
text = get_corpus()
(train_text, val_text, test_text), vocab = preprocess_text(
    text, val_size=0.1, test_size=0.1, min_count=MIN_COUNT, seed=0
    )
print("Done ({:.2f}s)".format(time() - t0))
print()

######################################################################
# training

model = RNNLM(len(vocab), EMBEDDING_DIM, HIDDEN_SIZE, SEQ_LEN)
dataloader = TrainValLoader(train_text, val_text, vocab, SEQ_LEN,
                            TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

print("Training...")
print("Length of training set: {}".format(len(dataloader.train.dataset)))
train_loss, val_loss = train(model, dataloader, optimizer, NUM_EPOCHS, device)

plot_loss(train_loss, val_loss)