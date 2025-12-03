import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from src.model import LSTMClassifier

DEVICE = "cpu"
EPOCHS = 2
LR = 2e-4
BATCH_SIZE = 16

# -------------------------
# 1 — Load dataset
# -------------------------
dataset = load_dataset("dair-ai/emotion")

# Convert 6 labels → 2 labels
positive_labels = {"joy", "love"}
negative_labels = {"anger", "sadness", "fear", "surprise"}

def convert_label(example):
    lbl_text = dataset["train"].features["label"].int2str(example["label"])
    return {"label_bin": 1 if lbl_text in positive_labels else 0}

dataset = dataset.map(convert_label)

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label_bin"]

# -------------------------
# 2 — Load tokenizer & BERT
# -------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
bert.eval()

# -------------------------
# 3 — LSTM Classifier
# -------------------------
model = LSTMClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# 4 — Batch generator
# -------------------------
def get_batch(start, end):
    texts = train_texts[start:end]
    labels = torch.tensor(train_labels[start:end], dtype=torch.long)

    tokens = tokenizer(
        texts,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=128
    )

    with torch.no_grad():
        embeddings = bert(
            tokens["input_ids"],
            attention_mask=tokens["attention_mask"]
        ).last_hidden_state

    return embeddings, labels


# -------------------------
# 5 — Training Loop
# -------------------------
num_samples = len(train_texts)

for epoch in range(EPOCHS):
    total_loss = 0

    for i in range(0, num_samples, BATCH_SIZE):
        emb, lbl = get_batch(i, i + BATCH_SIZE)

        logits = model(emb)
        loss = criterion(logits, lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss:.4f}")

# -------------------------
# 6 — Save model
# -------------------------
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
