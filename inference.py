import torch
from transformers import AutoTokenizer, AutoModel
from src.model import LSTMClassifier

DEVICE = "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
bert.eval()

model = LSTMClassifier()
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

def predict(text):
    tokens = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=128
    )

    with torch.no_grad():
        emb = bert(
            tokens["input_ids"],
            attention_mask=tokens["attention_mask"]
        ).last_hidden_state

        logits = model(emb)
        probs = torch.softmax(logits, dim=1)
        label = torch.argmax(probs).item()

    return "POSITIVE" if label == 1 else "NEGATIVE", float(probs[0][label])

if __name__ == "__main__":
    txt = "I really love this!"
    label, confidence = predict(txt)
    print(f"Prediction: {label} ({confidence:.4f})")
