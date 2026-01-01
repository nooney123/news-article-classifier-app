import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from datasets import load_dataset
from model_utils import TransformerBiLSTM, clean_text, LABELS
import numpy as np
import os
from tqdm import tqdm

# Config
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 1 # Reduced for faster training
SUB_SAMPLE = 0.2 # Use only 20% of data
LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "transformer_bilstm_best.pt")

def tokenize_function(examples, tokenizer):
    cleaned = [clean_text(t) for t in examples["text"]]
    return tokenizer(cleaned, padding="max_length", truncation=True, max_length=MAX_LEN)

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    
    # 1. Load Data
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    # Optimize: Subsample dataset
    if SUB_SAMPLE < 1.0:
        print(f"Subsampling dataset to {int(SUB_SAMPLE * 100)}%...")
        for split in dataset.keys():
            num_samples = int(len(dataset[split]) * SUB_SAMPLE)
            dataset[split] = dataset[split].select(range(num_samples))
    
    # AG News labels are 0-3, matching our LABELS list index
    # 0: World, 1: Sports, 2: Business, 3: Sci/Tech
    
    # 2. Tokenize
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=BATCH_SIZE)
    
    # 3. Model
    print("Initializing model...")
    model = TransformerBiLSTM(model_name=MODEL_NAME, num_labels=len(LABELS))
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # 4. Train Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        
        loop = tqdm(train_dataloader, leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            loop.set_description(f"Loss: {loss.item():.4f}")
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            print(f"Saving new best model to {CHECKPOINT_PATH}")
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    print("Training complete!")

if __name__ == "__main__":
    train()
