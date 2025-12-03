import torch
from transformers import AutoTokenizer, AutoModel
from src.model import LSTMClassifier

DEVICE = "cpu"

# Load tokenizer & BERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE)
bert.eval()

# Load LSTM model
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

    return label, float(probs[0][label])

# Test
if __name__ == "__main__":
    txt = "I really love this!"
    label, prob = predict(txt)
    print("Prediction:", "POSITIVE" if label==1 else "NEGATIVE", prob)
