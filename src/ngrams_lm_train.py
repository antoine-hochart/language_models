import os
import pickle
import numpy as np

from time import time
from tqdm import tqdm

from utils.data import get_corpus, preprocess_text
from utils.eval import get_perplexity, plot_perplexity
from models.ngram import SimpleInterpolNgrams

######################################################################
# load and preprocess text

print("Loading and preprocessing text...")
t0 = time()
text = get_corpus()
(train_text, val_text, test_text), _ = preprocess_text(
    text, val_size=0.1, test_size=0.1, min_count=5, seed=0
    )
print("Done ({:.2f}s)".format(time() - t0))

######################################################################
# train and evaluate language models

weight_range = np.linspace(0, 1, num=50, endpoint=False)
orders = [2, 3, 4]
scores = []

print("Evaluating Simple Interpolated N-gram models...")
for order in orders:
    print()
    print("Order {}".format(order))
    scores.append([])
    for w in tqdm(weight_range, ascii=True):
        lm = SimpleInterpolNgrams(train_text, order=order, weight=w)
        perplexity = get_perplexity(lm, val_text)
        if perplexity < min([p for perplex in scores for p in perplex], default=np.Inf):
            best_lm = lm
        scores[-1].append(perplexity)

######################################################################
# save best model

fpath = os.path.join(os.path.dirname(__file__), '..', 'data', 'models', 'ngram.pkl')
fpath = os.path.abspath(fpath)
os.makedirs(os.path.dirname(fpath), exist_ok=True)
with open(fpath, 'wb') as file:
    pickle.dump(best_lm, file)

print()
print("Best parameters: order {}, weight {}".format(best_lm.order, best_lm.weight))

######################################################################
# plot results

plot_perplexity(weight_range, scores, labels=['order {}'.format(o) for o in orders],
                title='perplexity of interpolated n-grams model - Proust')