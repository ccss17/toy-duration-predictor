import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
import os
import copy

# Import Optuna for hyperparameter optimization
import optuna

# --- 1. Configuration & Hyperparameters ---
# Data Parameters
REPO_ID = "ccss17/note-duration-dataset"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0

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

# --- 3. Data Preparation ---
class DurationDataset(Dataset):
    def __init__(self, processed_hf_dataset, model_type, mean, std):
        self.processed_dataset = processed_hf_dataset; self.model_type = model_type
        self.mean = mean; self.std = std
    def __len__(self): return len(self.processed_dataset)
    def __getitem__(self, idx):
        item = self.processed_dataset[idx]
        original_durations = torch.tensor(item['durations'], dtype=torch.float32)
        label = (torch.clamp(original_durations, 0, 1000) - self.mean) / (self.std + 1e-8)
        input_seq = torch.tensor(item['durations'] if self.model_type == 'A' else item['quantized_durations'], dtype=torch.long)
        singer_idx = torch.tensor(item['singer_idx'], dtype=torch.long)
        return {"input_seq": input_seq, "singer_id": singer_idx}, label

def prepare_data(model_type='B'):
    dataset = load_dataset(REPO_ID, split='train', trust_remote_code=True)
    unique_singer_ids = sorted(dataset.unique("singer_id"))
    singer_id_map = {sid: i for i, sid in enumerate(unique_singer_ids)}
    dataset = dataset.map(lambda ex: {"singer_idx": singer_id_map[ex["singer_id"]]})
    
    train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
    split_dataset = DatasetDict({'train': train_testvalid['train'], 'validation': test_valid['train'], 'test': test_valid['test']})

    all_training_durations = [item for sublist in split_dataset['train']['durations'] for item in sublist]
    durations_tensor = torch.clamp(torch.tensor(all_training_durations, dtype=torch.float32), 0, 1000)
    mean, std = durations_tensor.mean(), durations_tensor.std()
    
    def chunk_examples(examples):
        all_chunks = []
        for i in range(len(examples["durations"])):
            durs, q_durs, s_idx = examples["durations"][i], examples["quantized_durations"][i], examples["singer_idx"][i]
            for j in range(0, len(durs), SEQUENCE_LENGTH):
                d_chunk = durs[j:j+SEQUENCE_LENGTH]; q_chunk = q_durs[j:j+SEQUENCE_LENGTH]
                if len(d_chunk) < SEQUENCE_LENGTH:
                    padding = [PAD_TOKEN] * (SEQUENCE_LENGTH - len(d_chunk))
                    d_chunk.extend(padding); q_chunk.extend(padding)
                all_chunks.append({'durations': d_chunk, 'quantized_durations': q_chunk, 'singer_idx': s_idx})
        return {'chunks': all_chunks}

    processed_train = split_dataset['train'].map(chunk_examples, batched=True, remove_columns=split_dataset['train'].column_names)['chunks']
    processed_val = split_dataset['validation'].map(chunk_examples, batched=True, remove_columns=split_dataset['validation'].column_names)['chunks']
    processed_test = split_dataset['test'].map(chunk_examples, batched=True, remove_columns=split_dataset['test'].column_names)['chunks']
    
    train_ds = DurationDataset(processed_train, model_type, mean, std)
    val_ds = DurationDataset(processed_val, model_type, mean, std)
    test_ds = DurationDataset(processed_test, model_type, mean, std)

    return train_ds, val_ds, test_ds, mean, std, len(unique_singer_ids)

# --- 4. The Objective Function for Optuna ---
def objective(trial, model_type, train_ds, val_ds, num_singers, gpu_id):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    config = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "hidden_size": trial.suggest_categorical("hidden_size", [256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.2, 0.5),
        "singer_embedding_dim": trial.suggest_categorical("singer_embedding_dim", [16, 32])
    }
    
    model = ToyDurationPredictor(
        hidden_size=config["hidden_size"], num_layers=config["num_layers"],
        dropout=config["dropout"], num_singers=num_singers,
        singer_embedding_dim=config["singer_embedding_dim"]
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
            preds = model(input_seq, singer_id)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
            preds = model(input_seq, singer_id)
            val_loss += loss_fn(preds, labels).item()
    
    return val_loss / len(val_loader)

# --- 5. Main Orchestration Function ---
def run_experiment(model_type: str, gpu_id: int):
    print(f"\n{'='*20} STARTING EXPERIMENT FOR MODEL {model_type} ON GPU {gpu_id} {'='*20}")
    
    train_ds, val_ds, test_ds, mean, std, num_singers = prepare_data(model_type)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_type, train_ds, val_ds, num_singers, gpu_id), n_trials=10)
    
    best_config = study.best_params
    print(f"--- Best Hyperparameters for Model {model_type}: {best_config} ---")
    
    print(f"--- Training Final Model {model_type} with Best Config ---")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    final_model = ToyDurationPredictor(
        hidden_size=best_config["hidden_size"], num_layers=best_config["num_layers"],
        dropout=best_config["dropout"], num_singers=num_singers,
        singer_embedding_dim=best_config["singer_embedding_dim"]
    ).to(device)

    final_train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_config["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(30):
        final_model.train()
        for batch in tqdm(final_train_loader, desc=f"Final Training Epoch {epoch+1}/30"):
            inputs, labels = batch
            input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
            preds = final_model(input_seq, singer_id)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    model_save_path = f"./model_{model_type}.pth"
    torch.save(final_model.state_dict(), model_save_path)
    print(f"Final model for '{model_type}' saved to {model_save_path}")

    return model_save_path, best_config, test_ds, mean, std

# --- NEW: Function for Final Evaluation ---
def evaluate_and_compare(model_A_path, config_A, model_B_path, config_B, gpu_id=0):
    print(f"\n{'='*20} FINAL MODEL COMPARISON {'='*20}")
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # Load Model A with its best config
    model_A = ToyDurationPredictor(
        hidden_size=config_A["hidden_size"], num_layers=config_A["num_layers"],
        dropout=config_A["dropout"], num_singers=18,
        singer_embedding_dim=config_A["singer_embedding_dim"]
    ).to(device)
    model_A.load_state_dict(torch.load(model_A_path, map_location=device))
    model_A.eval()

    # Load Model B with its best config
    model_B = ToyDurationPredictor(
        hidden_size=config_B["hidden_size"], num_layers=config_B["num_layers"],
        dropout=config_B["dropout"], num_singers=18,
        singer_embedding_dim=config_B["singer_embedding_dim"]
    ).to(device)
    model_B.load_state_dict(torch.load(model_B_path, map_location=device))
    model_B.eval()

    # Get the test loader with 'quantized' input and the normalization stats
    _, _, test_ds, mean, std, _ = prepare_data(model_type='B')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    total_mae_A, total_mae_B, total_samples = 0, 0, 0
    print("Evaluating models on the test set...")
    for batch in tqdm(test_loader, desc="Final Evaluation"):
        inputs, labels_norm = batch
        input_seq, singer_id = inputs['input_seq'].to(device), inputs['singer_id'].to(device)
        labels_norm = labels_norm.to(device)
        
        with torch.no_grad():
            preds_A_norm = model_A(input_seq, singer_id)
            preds_B_norm = model_B(input_seq, singer_id)

            # De-normalize predictions and labels for MAE calculation
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

# --- Main Execution Block ---
if __name__ == "__main__":
    # Run the full experiment for both models sequentially
    model_a_path, config_a, _, _, _ = run_experiment(model_type='A', gpu_id=0)
    model_b_path, config_b, test_ds_B, mean_B, std_B = run_experiment(model_type='B', gpu_id=1)

    # Run the final comparison
    evaluate_and_compare(model_a_path, config_a, model_b_path, config_b, gpu_id=0)
