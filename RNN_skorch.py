from dstoolbox.transformers import Padder2d
from dstoolbox.transformers import TextFeaturizer
import numpy as np
from scipy import stats
from sklearn.datasets import load_files
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from skorch import NeuralNetClassifier
import torch
from torch import nn
F = nn.functional

from data_preprocessing import preprocessing
Dp_training = preprocessing()

from sklearn.utils import shuffle
import pandas as pd

np.random.seed(0)
torch.manual_seed(0)
# torch.cuda.manual_seed(0)

VOCAB_SIZE = 1000  # This is on the low end
MAX_LEN = 50  # Texts are pretty long on average, this is on the low end
USE_CUDA = torch.cuda.is_available()  # Set this to False if you don't want to use CUDA
NUM_CV_STEPS = 10  # Number of randomized search steps to perform

steps = [
    ('to_idx', TextFeaturizer(max_features=VOCAB_SIZE)),
    ('pad', Padder2d(max_len=MAX_LEN, pad_value=VOCAB_SIZE, dtype=int)),
]




downsampled_data = pd.read_csv('./downsampled_data.csv')
downsampled_data= shuffle(downsampled_data, random_state=0)

#get training_text_ints
text_integers_training = Dp_training.text_ints_extract(downsampled_data)


seq_length = 50
training_features = Dp_training.pad_features(text_integers_training, seq_length)
training_labels = Dp_training.get_labels(downsampled_data.LABEL)

X = downsampled_data.punctuation_out
y = training_labels

# pipe = Pipeline(steps).fit_transform(X[:3])
# print(pipe)

class RNNClassifier(nn.Module):
    def __init__(
            self,
            embedding_dim=128,
            rec_layer_type='lstm',
            num_units=128,
            num_layers=2,
            dropout=0,
            bidirectional=False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rec_layer_type = rec_layer_type.lower()
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.reset_weights()

    def reset_weights(self):
        self.emb = nn.Embedding(VOCAB_SIZE + 1, embedding_dim=self.embedding_dim)

        rec_layer = {'lstm': nn.LSTM, 'gru': nn.GRU}[self.rec_layer_type]
        # We have to make sure that the recurrent layer is batch_first,
        # since sklearn assumes the batch dimension to be the first
        self.rec = rec_layer(
            self.embedding_dim,
            self.num_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        self.drop = nn.Dropout(self.dropout)
        self.output = nn.Linear(self.num_units, 2)

    def forward(self, X):
        embeddings = self.emb(X)

        # from the recurrent layer, only take the activities from the last sequence step
        if self.rec_layer_type == 'gru':
            _, rec_out = self.rec(embeddings)
        else:
            _, (rec_out, _) = self.rec(embeddings)
        rec_out = rec_out[-1]  # take output of last RNN layer

        drop = self.drop(rec_out)
        # Remember that the final non-linearity should be softmax, so that our predict_proba
        # method outputs actual probabilities!
        out = F.softmax(self.output(drop), dim=-1)
        return out

net = NeuralNetClassifier(
    RNNClassifier,
    device=('cuda' if USE_CUDA else 'cpu'),
    max_epochs=5,
    lr=0.01,
    optimizer=torch.optim.RMSprop,
)

pipe = Pipeline(steps + [('net', net)])
# pipe.fit(X, y)

pipe.set_params(net__verbose=0, net__train_split=None)


params = {
    'to_idx__stop_words': ['english', None],
    'to_idx__lowercase': [False, True],
    'to_idx__ngram_range': [(1, 1), (2, 2)],
    'net__module__embedding_dim': stats.randint(32, 256 + 1),
    'net__module__rec_layer_type': ['gru', 'lstm'],
    'net__module__num_units': stats.randint(32, 256 + 1),
    'net__module__num_layers': [1, 2, 3],
    'net__module__dropout': stats.uniform(0, 0.9),
    'net__module__bidirectional': [True, False],
    'net__lr': [10**(-stats.uniform(1, 5).rvs()) for _ in range(NUM_CV_STEPS)],
    'net__max_epochs': [5, 10],

}


search = RandomizedSearchCV(
    pipe, params, n_iter=NUM_CV_STEPS, verbose=2, refit=False, scoring='accuracy', cv=5,return_train_score=True)

search.fit(X, y)

print(search.best_score_)
print()
print(search.best_params_)