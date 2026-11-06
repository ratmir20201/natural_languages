import torch
import torch.nn as nn

text = "deep learning is the future of natural language processing"
tokens = text.lower().split()

vocab = {"<unk>": 0}
for token in tokens:
    vocab.setdefault(token, len(vocab))

encoded = torch.tensor([vocab.get(token, 0) for token in tokens])
print(encoded)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)


model = LSTMModel(len(vocab), 32, 64)
output = model(encoded.unsqueeze(0))
print(output.shape)
