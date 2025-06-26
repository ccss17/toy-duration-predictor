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

# Import Ray and the correct Tune modules
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch

# Import Optuna itself for defining the search space
import optuna

# --- 1. Configuration & Hyperparameters ---
# Data Parameters
REPO_ID = "ccss17/note-duration-dataset"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0
BATCH_SIZE = 64

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


# --- 3. Data Preparation (Remains the same) ---
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
        # This function processes a batch of songs into fixed-length chunks
        all_chunks = []
        for i in range(len(examples["durations"])):
            durs, q_durs, s_idx = examples["durations"][i], examples["quantized_durations"][i], examples["singer_idx"][i]
            for j in range(0, len(durs), SEQUENCE_LENGTH):
                d_chunk = durs[j:j+SEQUENCE_LENGTH]; q_chunk = q_durs[j:j+SEQUENCE_LENGTH]
                if len(d_chunk) < SEQUENCE_LENGTH:
                    padding = [PAD_TOKEN] * (SEQUENCE_LENGTH - len(d_chunk))
                    d_chunk.extend(padding); q_chunk.extend(padding)
                all_chunks.append({'durations': d_chunk, 'quantized_durations': q_chunk, 'singer_idx': s_idx})
        return all_chunks
    
    processed_train = split_dataset['train'].map(chunk_examples, batched=True, remove_columns=split_dataset['train'].column_names)
    processed_val = split_dataset['validation'].map(chunk_examples, batched=True, remove_columns=split_dataset['validation'].column_names)
    processed_test = split_dataset['test'].map(chunk_examples, batched=True, remove_columns=split_dataset['test'].column_names)
    
    train_ds = DurationDataset(processed_train, model_type, mean, std)
    val_ds = DurationDataset(processed_val, model_type, mean, std)
    test_ds = DurationDataset(processed_test, model_type, mean, std)

    return train_ds, val_ds, test_ds, mean, std


# --- 4. The Objective Function for Optuna/Ray Tune ---
# This function defines a single training and validation run for ONE trial.
def objective_function(config, model_type, train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with hyperparameters from the trial's config
    model = ToyDurationPredictor(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        num_singers=18,
        singer_embedding_dim=config["singer_embedding_dim"]
    ).to(device)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.MSELoss()

    for epoch in range(15): # Train for a fixed number of epochs per trial
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
            preds = model(input_seq, singer_id)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
                preds = model(input_seq, singer_id)
                val_loss += loss_fn(preds, labels).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Report the validation loss back to Ray Tune
        tune.report(val_loss=avg_val_loss)


# --- 5. Main Orchestration Function ---
def run_experiment(model_type: str, gpu_id: int):
    print(f"\n{'='*20} STARTING EXPERIMENT FOR MODEL {model_type} ON GPU {gpu_id} {'='*20}")

    # Prepare data once
    train_ds, val_ds, test_ds, mean, std = prepare_data(model_type)

    # Define the search space using Optuna's suggestion API
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "batch_size": tune.choice([32, 64, 128]),
        "hidden_size": tune.choice([256, 512]),
        "num_layers": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0.2, 0.5),
        "singer_embedding_dim": tune.choice([16, 32])
    }

    # Define the Optuna search algorithm
    optuna_search = OptunaSearch(metric="val_loss", mode="min")

    # Create the trainable function, passing the datasets
    trainable_with_params = tune.with_parameters(objective_function, model_type=model_type, train_dataset=train_ds, val_dataset=val_ds)
    
    # Run the tuning job
    analysis = tune.run(
        trainable_with_params,
        config=search_space,
        search_alg=optuna_search,
        num_samples=20, # Number of trials to run
        resources_per_trial={"cpu": 2, "gpu": 1},
        name=f"optuna_tune_model_{model_type}"
    )

    best_config = analysis.best_config
    print(f"--- Best Hyperparameters for Model {model_type}: {best_config} ---")
    
    # Now train the final model with the best found hyperparameters
    print(f"--- Training Final Model {model_type} with Best Config ---")
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    final_model = ToyDurationPredictor(
        hidden_size=best_config["hidden_size"], num_layers=best_config["num_layers"],
        dropout=best_config["dropout"], num_singers=18,
        singer_embedding_dim=best_config["singer_embedding_dim"]
    ).to(device)

    final_train_loader = DataLoader(train_ds, batch_size=best_config["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_config["lr"])
    loss_fn = nn.MSELoss()

    # Train for a longer duration on the full dataset
    for epoch in range(30): # e.g., train for 30 epochs
        final_model.train()
        for batch in tqdm(final_train_loader, desc=f"Final Training Epoch {epoch+1}/30"):
             inputs, labels = batch
             input_seq, singer_id, labels = inputs['input_seq'].to(device), inputs['singer_id'].to(device), labels.to(device)
             preds = final_model(input_seq, singer_id)
             loss = loss_fn(preds, labels)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

    model_save_path = f"./model_{model_type}.pth"
    torch.save(final_model.state_dict(), model_save_path)
    print(f"Final model for '{model_type}' saved to {model_save_path}")

    return model_save_path, test_ds, mean, std

# --- Main Execution Block ---
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, num_gpus=2) # Initialize Ray for 2 GPUs
    
    # --- Run experiments in parallel ---
    # We use remote tasks to run both experiments at the same time on different GPUs
    @ray.remote(num_gpus=1)
    def run_experiment_task(model_type, gpu_id):
        return run_experiment(model_type, gpu_id)

    result_ref_A = run_experiment_task.remote('A', 0)
    result_ref_B = run_experiment_task.remote('B', 1)

    model_a_path, test_ds_A, mean_A, std_A = ray.get(result_ref_A)
    model_b_path, test_ds_B, mean_B, std_B = ray.get(result_ref_B)

    # --- Final Evaluation and Comparison would go here ---
    # This part would load the two saved models and compare them on the test set
    # as shown in the previous vanilla PyTorch script.
    print("\nAll experiments finished. Ready for final evaluation.")
    
    ray.shutdown()
