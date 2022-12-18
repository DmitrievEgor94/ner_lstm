from torch import nn
from torch.nn import functional as F


class NerLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tags_dim):
        super(NerLSTM, self).__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.last_layer = nn.Linear(2*hidden_dim, tags_dim)

    def forward(self, x):
        x = self.embedding_layer(x)

        x, _ = self.bi_lstm(x)

        x = self.last_layer(x)
        out = F.softmax(x, dim=-1)

        return out