import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # For progress bars
import os
import copy

# --- 1. Configuration & Fixed Hyperparameters ---
# Data Parameters
REPO_ID = "ccss17/note-duration-dataset"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0
BATCH_SIZE = 64

# --- MODIFIED: A single, more powerful configuration for BOTH models ---
# We use the same architecture for both Model A and B for a fair comparison.
# We increase the capacity from the original because we have evidence the smaller
# model was not powerful enough for the more difficult task of Model B.
MODEL_CONFIG = {
    "num_singers": 18,
    "singer_embedding_dim": 32,
    "hidden_size": 256,
    "num_layers": 3,
    "dropout": 0.4
}

# Training Parameters
LEARNING_RATE = 1e-4 
NUM_EPOCHS = 50 
EARLY_STOPPING_PATIENCE = 4
GPU_ID = 0 

# --- 2. Model Architecture Definition (Vanilla PyTorch) ---
class ToyDurationPredictor(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, num_singers, singer_embedding_dim):
        super().__init__()
        self.singer_embedding = nn.Embedding(num_singers, singer_embedding_dim)
        self.rnn = nn.GRU(
            input_size=1 + singer_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x_seq, x_sid):
        x_seq = x_seq.unsqueeze(-1).float()
        sid_emb = self.singer_embedding(x_sid)
        sid_emb_expanded = sid_emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        combined_input = torch.cat([x_seq, sid_emb_expanded], dim=-1)
        outputs, _ = self.rnn(combined_input)
        prediction = self.fc(outputs)
        return prediction.squeeze(-1)


# --- 3. Data Preparation (Custom PyTorch Dataset) ---
class DurationDataset(Dataset):
    """A standard PyTorch Dataset for our chunked data."""
    def __init__(self, processed_hf_dataset, model_type, mean, std):
        self.processed_dataset = processed_hf_dataset
        self.model_type = model_type
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.processed_dataset)

    def __getitem__(self, idx):
        item = self.processed_dataset[idx]
        original_durations = torch.tensor(item['durations'], dtype=torch.float32)
        label = (torch.clamp(original_durations, 0, 1000) - self.mean) / (self.std + 1e-8)
        
        if self.model_type == 'A':
            input_seq = torch.tensor(item['durations'], dtype=torch.long)
        else:
            input_seq = torch.tensor(item['quantized_durations'], dtype=torch.long)
            
        singer_idx = torch.tensor(item['singer_idx'], dtype=torch.long)
        
        return {"input_seq": input_seq, "singer_id": singer_idx}, label

def prepare_data(model_type='B'):
    """Loads, splits, and prepares data, returning DataLoaders."""
    print(f"--- Preparing data for Model {model_type} ---")
    dataset = load_dataset(REPO_ID, split='train', trust_remote_code=True)
    
    unique_singer_ids = sorted(dataset.unique("singer_id"))
    singer_id_map = {sid: i for i, sid in enumerate(unique_singer_ids)}
    
    def map_singer_ids(example):
        example["singer_idx"] = singer_id_map[example["singer_id"]]
        return example
    dataset = dataset.map(map_singer_ids)
    
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    split_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

    all_training_durations = [item for sublist in split_dataset['train']['durations'] for item in sublist]
    durations_tensor = torch.tensor(all_training_durations, dtype=torch.float32)
    durations_tensor = torch.clamp(durations_tensor, 0, 1000)
    mean = durations_tensor.mean()
    std = durations_tensor.std()
    print(f"Calculated training set stats: Mean={mean:.2f}, Std={std:.2f}")
    
    def chunk_examples_batched(examples):
        chunked_output = {'durations': [], 'quantized_durations': [], 'singer_idx': []}
        for i in range(len(examples["durations"])):
            durs, q_durs, s_idx = examples["durations"][i], examples["quantized_durations"][i], examples["singer_idx"][i]
            for j in range(0, len(durs), SEQUENCE_LENGTH):
                d_chunk, q_chunk = durs[j:j+SEQUENCE_LENGTH], q_durs[j:j+SEQUENCE_LENGTH]
                if len(d_chunk) < SEQUENCE_LENGTH:
                    padding = [PAD_TOKEN] * (SEQUENCE_LENGTH - len(d_chunk))
                    d_chunk.extend(padding); q_chunk.extend(padding)
                chunked_output['durations'].append(d_chunk)
                chunked_output['quantized_durations'].append(q_chunk)
                chunked_output['singer_idx'].append(s_idx)
        return chunked_output

    processed_train = split_dataset['train'].map(chunk_examples_batched, batched=True, remove_columns=split_dataset['train'].column_names)
    processed_val = split_dataset['validation'].map(chunk_examples_batched, batched=True, remove_columns=split_dataset['validation'].column_names)
    processed_test = split_dataset['test'].map(chunk_examples_batched, batched=True, remove_columns=split_dataset['test'].column_names)
    
    train_ds = DurationDataset(processed_train, model_type, mean, std)
    val_ds = DurationDataset(processed_val, model_type, mean, std)
    test_ds = DurationDataset(processed_test, model_type, mean, std)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)
    
    return train_loader, val_loader, test_loader, singer_id_map, mean, std

# --- 4. The "Vanilla" Training Loop ---
def train_model(model_type='B', gpu_id=0):
    print(f"\n{'='*20} TRAINING MODEL {model_type} ON GPU {gpu_id} {'='*20}")
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, _, _, _, _ = prepare_data(model_type)
    
    # Use the single, more powerful config for both models
    config = MODEL_CONFIG
    
    model = ToyDurationPredictor(
        hidden_size=config["hidden_size"], num_layers=config["num_layers"],
        dropout=config["dropout"], num_singers=config["num_singers"],
        singer_embedding_dim=config["singer_embedding_dim"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    # Define the save path once at the beginning
    model_save_path = f"./model_{model_type}.pth"
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            inputs, labels = batch
            input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
            predictions = model(input_seq, singer_id)
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
                predictions = model(input_seq, singer_id)
                val_loss += loss_fn(predictions, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

        # --- MODIFIED: Save the model to a file every time we find a better one ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the new best model state immediately
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model found! Checkpoint saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Stopping early after {patience_counter} epochs with no improvement.")
            break
    
    print(f"Training finished. Best model for '{model_type}' is at {model_save_path}")
    return model_save_path

# --- 5. Final Evaluation and Comparison ---
def evaluate_and_compare(model_A_path, model_B_path, gpu_id=0):
    print(f"\n{'='*20} FINAL MODEL COMPARISON {'='*20}")
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")
    print(f"Using device for evaluation: {device}")

    # Use the single config for both models
    config = MODEL_CONFIG
    model_A = ToyDurationPredictor(**config).to(device)
    model_A.load_state_dict(torch.load(model_A_path, map_location=device))
    model_A.eval()

    model_B = ToyDurationPredictor(**config).to(device)
    model_B.load_state_dict(torch.load(model_B_path, map_location=device))
    model_B.eval()

    _, _, test_loader, _, mean, std = prepare_data(model_type='B')
    
    total_mae_A, total_mae_B, total_samples = 0, 0, 0
    print("Evaluating models on the test set...")
    for batch in tqdm(test_loader, desc="Final Evaluation"):
        inputs, labels_norm = batch
        input_seq, singer_id, labels_norm = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels_norm.to(device)
        
        with torch.no_grad():
            preds_A_norm = model_A(input_seq, singer_id)
            preds_B_norm = model_B(input_seq, singer_id)

            preds_A = (preds_A_norm * std) + mean
            preds_B = (preds_B_norm * std) + mean
            original_labels = (labels_norm * std) + mean
            
        total_mae_A += nn.functional.l1_loss(preds_A, original_labels, reduction='sum').item()
        total_mae_B += nn.functional.l1_loss(preds_B, original_labels, reduction='sum').item()
        total_samples += original_labels.numel()
    
    avg_mae_A = total_mae_A / total_samples
    avg_mae_B = total_mae_B / total_samples

    print(f"\nFinal Test MAE for Model A (Control): {avg_mae_A:.4f} ticks")
    print(f"Final Test MAE for Model B (Your Method): {avg_mae_B:.4f} ticks")

    # --- Plotting and Final Results ---
    print("\n--- Comparison Plot ---")
    batch = next(iter(test_loader))
    inputs, labels_norm = batch
    
    with torch.no_grad():
        preds_A_norm = model_A(inputs['input_seq'].to(device), inputs['singer_id'].to(device))[0]
        preds_B_norm = model_B(inputs['input_seq'].to(device), inputs['singer_id'].to(device))[0]
        
        preds_A = ((preds_A_norm * std) + mean).cpu().numpy()
        preds_B = ((preds_B_norm * std) + mean).cpu().numpy()

    input_vals = inputs['input_seq'][0].cpu().numpy()
    ground_truth_vals = ((labels_norm[0] * std) + mean).cpu().numpy()

    plt.figure(figsize=(18, 7))
    plt.plot(input_vals, label="Input (Quantized)", linestyle='--', alpha=0.6, color='gray')
    plt.plot(preds_A, label=f"Model A Prediction (MAE: {avg_mae_A:.2f})", linewidth=2, marker='x', alpha=0.8)
    plt.plot(preds_B, label=f"Model B Prediction (MAE: {avg_mae_B:.2f})", linewidth=2, marker='o', alpha=0.8, markersize=4)
    plt.plot(ground_truth_vals, label="Ground Truth (Original)", linestyle=':', color='black', linewidth=2)
    plt.title(f"Model Comparison for Singer Index: {inputs['singer_id'][0].item()}")
    plt.xlabel("Note Index in Sequence"); plt.ylabel("Duration (ticks)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()


# --- Main Execution Block ---
if __name__ == "__main__":
    # model_a_path = train_model(model_type='A', gpu_id=0)
    model_b_path = train_model(model_type='B', gpu_id=1)
    
    # # We still need the final config to load the model architecture for evaluation

    # model_a_path = "./model_A.pth"
    # model_b_path = "./model_B.pth"
    # evaluate_and_compare(model_a_path, model_b_path, gpu_id=0)
