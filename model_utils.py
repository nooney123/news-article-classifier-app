import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import re, emoji

# Labels for AG News dataset
LABELS = ["World", "Sports", "Business", "Sci/Tech"]

# Cleaner
url_pat = re.compile(r"https?://\S+|www\.\S+")
mention_pat = re.compile(r"@\w+")
hashtag_pat = re.compile(r"#(\w+)")
extra_space = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not isinstance(t, str): return ""
    t = url_pat.sub(" ", t)
    t = mention_pat.sub(" ", t)
    t = hashtag_pat.sub(lambda m: m.group(1), t)
    t = emoji.demojize(t, language="en")
    t = extra_space.sub(" ", t).strip()
    return t

# Model
class TransformerBiLSTM(nn.Module):
    def __init__(self, model_name, num_labels, finetune_transformer=True, lstm_hidden_ratio=0.5, dropout=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        if not finetune_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False
        lstm_hidden = int(hidden_size * lstm_hidden_ratio)
        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_labels)

    def masked_mean(self, x, mask):
        mask = mask.unsqueeze(-1).type_as(x)
        x = x * mask
        summed = x.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state
        lstm_out, _ = self.bilstm(seq)
        pooled = self.masked_mean(lstm_out, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        return logits

# Load model + tokenizer
def load_model(checkpoint_path="checkpoints/transformer_bilstm_best.pt", model_name="distilbert-base-uncased", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = TransformerBiLSTM(model_name=model_name, num_labels=len(LABELS))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return tokenizer, model

# Prediction function
def predict_headline(text, tokenizer, model, device="cpu", max_len=32, topk=3):
    cleaned = clean_text(text)
    enc = tokenizer(cleaned, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    top_idx = probs.argsort()[::-1][:topk]
    return [(LABELS[i], float(probs[i])) for i in top_idx]
