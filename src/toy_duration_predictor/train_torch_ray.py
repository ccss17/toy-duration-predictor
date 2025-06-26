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

# Import Ray and the Ray Train components
import ray
import ray.train.torch
from ray.train import ScalingConfig, RunConfig, Checkpoint, get_dataset_shard

# --- 1. Configuration & Hyperparameters ---
REPO_ID = "ccss17/note-duration-dataset"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0
BATCH_SIZE = 64 # This will now be the batch size PER WORKER

# Unified Model Config for a fair comparison
MODEL_CONFIG = {
    "num_singers": 18,
    "singer_embedding_dim": 32,
    "hidden_size": 512,
    "num_layers": 4,
    "dropout": 0.4,
    "lr": 1e-4
}

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

# --- 3. Data Preparation for Ray Data ---
def prepare_ray_datasets(model_type='B'):
    print(f"--- Preparing Ray Datasets for Model {model_type} ---")
    hf_dataset = load_dataset(REPO_ID, split='train', trust_remote_code=True)
    
    unique_singer_ids = sorted(hf_dataset.unique("singer_id"))
    singer_id_map = {sid: i for i, sid in enumerate(unique_singer_ids)}
    
    def map_and_chunk(batch):
        new_batch = {'input_seq': [], 'label': [], 'singer_idx': []}
        for i in range(len(batch["durations"])):
            durs, q_durs, sid = batch["durations"][i], batch["quantized_durations"][i], batch["singer_id"][i]
            s_idx = singer_id_map[sid]
            input_source = durs if model_type == 'A' else q_durs
            
            for j in range(0, len(durs), SEQUENCE_LENGTH):
                d_chunk, i_chunk = durs[j:j+SEQUENCE_LENGTH], input_source[j:j+SEQUENCE_LENGTH]
                if len(d_chunk) < SEQUENCE_LENGTH:
                    padding = [PAD_TOKEN] * (SEQUENCE_LENGTH - len(d_chunk))
                    d_chunk.extend(padding); i_chunk.extend(padding)
                new_batch['input_seq'].append(i_chunk)
                new_batch['label'].append(d_chunk)
                new_batch['singer_idx'].append(s_idx)
        return new_batch

    ray_ds = ray.data.from_huggingface(hf_dataset)
    processed_ds = ray_ds.flat_map(map_and_chunk)
    train_ds, val_ds = processed_ds.train_test_split(test_size=0.2, seed=42)
    return {"train": train_ds, "validation": val_ds}

# --- 4. The Ray Train Training Loop ---
def train_loop_per_worker(config):
    lr, epochs, model_params = config["lr"], config["epochs"], config["model_config"]
    
    # Get data shard for this worker and create dataloader
    train_ds = get_dataset_shard("train")
    train_dataloader = train_ds.iter_torch_batches(batch_size=BATCH_SIZE, dtypes={"input_seq": torch.long, "label": torch.float32, "singer_idx": torch.long})

    model = ToyDurationPredictor(**model_params)
    model = ray.train.torch.prepare_model(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            input_seq, singer_idx, labels = batch['input_seq'], batch['singer_idx'], batch['label']
            preds = model(input_seq, singer_idx)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Ray Train handles validation and checkpointing in a simplified way
        # For simplicity, we report final loss. For early stopping, use a validation set.
        checkpoint = Checkpoint.from_dict(dict(epoch=epoch, model_weights=model.module.state_dict()))
        ray.train.report({"loss": loss.item()}, checkpoint=checkpoint)

# --- 5. NEW: Wrapper Function for Parallel Training ---
# This function will be launched as a remote task, one for each model type.
@ray.remote(num_gpus=1)
def run_and_train_model(model_type: str):
    print(f"\n{'='*20} LAUNCHING TRAINING FOR MODEL {model_type} {'='*20}")
    
    datasets, _ = prepare_ray_datasets(model_type=model_type)
    
    trainer = ray.train.torch.TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "lr": MODEL_CONFIG["lr"],
            "epochs": 15 if model_type == 'A' else 30, # Different epochs
            "model_config": MODEL_CONFIG
        },
        scaling_config=ScalingConfig(
            num_workers=1,  # Number of GPUs to use for this one job
            use_gpu=True
        ),
        datasets=datasets,
    )
    result = trainer.fit()
    best_checkpoint = result.get_best_checkpoint(metric="loss", mode="min")
    print(f"Best checkpoint for Model {model_type}: {best_checkpoint}")
    return best_checkpoint

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # Ensure Ray is initialized
    ray.init(ignore_reinit_error=True)
    
    # --- MODIFIED: Launch both training jobs in parallel ---
    print("Launching training for Model A and Model B in parallel on 2 GPUs...")
    
    # These calls are non-blocking. They start the tasks and return immediately.
    result_ref_A = run_and_train_model.remote('A')
    result_ref_B = run_and_train_model.remote('B')
    
    # ray.get() blocks until both remote tasks are finished and gets their results.
    best_checkpoint_A, best_checkpoint_B = ray.get([result_ref_A, result_ref_B])
    
    print("\n" + "="*20 + " ALL TRAINING COMPLETE " + "="*20)
    print(f"Final Checkpoint for Model A: {best_checkpoint_A}")
    print(f"Final Checkpoint for Model B: {best_checkpoint_B}")

    # --- Final Evaluation and Comparison would go here ---
    # You would now load the models from these two checkpoints and run the
    # evaluate_and_compare function as before.
    
    ray.shutdown()
