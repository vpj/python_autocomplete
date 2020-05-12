from torch import nn


class SimpleLstmModel(nn.Module):
    def __init__(self, *,
                 encoding_size,
                 embedding_size,
                 lstm_size,
                 lstm_layers):
        super().__init__()

        self.embedding = nn.Embedding(encoding_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_size,
                            num_layers=lstm_layers)
        self.fc = nn.Linear(lstm_size, encoding_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, h0, c0):
        # shape of x is [seq, batch, feat]
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        logits = self.fc(out)

        return self.softmax(logits), logits, (hn, cn)
