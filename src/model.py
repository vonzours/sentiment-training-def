import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, embeddings):
        lstm_out, _ = self.lstm(embeddings)
        return self.fc(lstm_out[:, -1, :])
