import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModel
from src.model import LSTMClassifier

# -------- CONFIG --------
EPOCHS = 2
LR = 2e-4
DEVICE = "cpu"

# EXEMPLE minimal dataset (à remplacer par ton vrai dataset)
texts = [
    "I love this!", 
    "This is amazing!", 
    "I hate this.",
    "This is terrible."
]
labels = [1, 1, 0, 0]

# -------- LOAD BERT --------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
bert.eval()

# -------- MODEL --------
model = LSTMClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def get_embeddings(text):
    tokens = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=128
    )
    with torch.no_grad():
        out = bert(
            tokens["input_ids"].to(DEVICE),
            attention_mask=tokens["attention_mask"].to(DEVICE)
        ).last_hidden_state
    return out

# -------- TRAIN LOOP --------
for epoch in range(EPOCHS):
    total_loss = 0
    for text, label in zip(texts, labels):
        emb = get_embeddings(text)
        logits = model(emb)
        loss = criterion(logits, torch.tensor([label]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# -------- SAVE MODEL --------
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
