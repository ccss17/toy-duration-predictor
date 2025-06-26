import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

# Import Ray and Tune for hyperparameter search
from ray import tune
# --- THE FIX IS HERE: Import the new, correct callback ---
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

# Import tools for demos
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration & Hyperparameters ---
# Data Parameters
REPO_ID = "ccss17/note-duration-dataset"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0
BATCH_SIZE = 64

# Model Parameters (defaults, will be overridden by Tune)
SINGER_EMBEDDING_DIM = 16
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3

# Configuration for wandb
WANDB_ENABLED = True
WANDB_ENTITY = "ccss17" # Your wandb username
WANDB_PROJECT = "toy-duration-predictor-lightning"


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


# --- 3. Data Preparation (PyTorch Dataset & Lightning DataModule) ---
class DurationDataset(Dataset):
    def __init__(self, processed_hf_dataset, model_type='B'):
        self.data = processed_hf_dataset
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = torch.tensor(item['durations'], dtype=torch.float32)
        
        if self.model_type == 'A':
            input_seq = torch.tensor(item['durations'], dtype=torch.long)
        else:
            input_seq = torch.tensor(item['quantized_durations'], dtype=torch.long)
            
        singer_idx = torch.tensor(item['singer_idx'], dtype=torch.long)
        
        return {"input_seq": input_seq, "singer_id": singer_idx}, label

class DurationDataModule(pl.LightningDataModule):
    def __init__(self, model_type='B', batch_size=32):
        super().__init__()
        self.model_type = model_type
        self.batch_size = batch_size
        self.singer_id_map = {}
        self.num_singers = 0

    def setup(self, stage=None):
        dataset = load_dataset(REPO_ID, split='train', trust_remote_code=True)
        unique_singer_ids = sorted(dataset.unique("singer_id"))
        self.num_singers = len(unique_singer_ids)
        self.singer_id_map = {sid: i for i, sid in enumerate(unique_singer_ids)}
        
        def map_singer_ids(example):
            example["singer_idx"] = self.singer_id_map[example["singer_id"]]
            return example
            
        dataset = dataset.map(map_singer_ids)
        
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)
        self.split_dataset = DatasetDict({
            'train': train_test_split['train'],
            'valid': test_valid_split['train'],
            'test': test_valid_split['test']
        })

        def chunk_examples(examples):
            chunked = {'durations': [], 'quantized_durations': [], 'singer_idx': []}
            for i in range(len(examples["durations"])):
                durs, q_durs, s_idx = examples["durations"][i], examples["quantized_durations"][i], examples["singer_idx"][i]
                for j in range(0, len(durs), SEQUENCE_LENGTH):
                    d_chunk, q_chunk = durs[j:j+SEQUENCE_LENGTH], q_durs[j:j+SEQUENCE_LENGTH]
                    if len(d_chunk) < SEQUENCE_LENGTH:
                        padding = [PAD_TOKEN] * (SEQUENCE_LENGTH - len(d_chunk))
                        d_chunk.extend(padding); q_chunk.extend(padding)
                    chunked['durations'].append(d_chunk); chunked['quantized_durations'].append(q_chunk); chunked['singer_idx'].append(s_idx)
            return chunked

        processed_splits = self.split_dataset.map(chunk_examples, batched=True, remove_columns=dataset.column_names)

        self.train_ds = DurationDataset(processed_splits['train'], self.model_type)
        self.val_ds = DurationDataset(processed_splits['valid'], self.model_type)
        self.test_ds = DurationDataset(processed_splits['test'], self.model_type)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=4, persistent_workers=True)


# --- 4. The LightningModule ---
class LightningTDP(pl.LightningModule):
    def __init__(self, num_singers, learning_rate, hidden_size, num_layers, dropout):
        super().__init__()
        self.save_hyperparameters()
        self.model = ToyDurationPredictor(
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
            num_singers=num_singers, singer_embedding_dim=SINGER_EMBEDDING_DIM
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, batch): return self.model(batch['input_seq'], batch['singer_id'])
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs); loss = self.loss_fn(preds, labels)
        self.log('train_loss', loss); return loss
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs); loss = self.loss_fn(preds, labels)
        self.log('val_loss', loss)
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs); loss = self.loss_fn(preds, labels)
        mae = nn.functional.l1_loss(preds, labels)
        self.log('test_loss', loss); self.log('test_mae', mae)
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- 5. Training Function for Ray Tune ---
def train_for_tune(config, datamodule):
    model = LightningTDP(
        num_singers=datamodule.num_singers,
        learning_rate=config["lr"], hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], dropout=config["dropout"]
    )
    
    # --- THE FIX IS HERE: Use the new callback ---
    # It reports metrics and also handles checkpointing for Ray Tune.
    tune_callback = TuneReportCheckpointCallback(
        metrics={"val_loss": "val_loss"}, 
        filename="tune_ckpt", 
        on="validation_end"
    )

    trainer = pl.Trainer(
        max_epochs=config["epochs"], 
        enable_progress_bar=False, 
        logger=False,
        accelerator="auto", 
        devices=1,
        callbacks=[tune_callback] # Use the new callback
    )
    trainer.fit(model, datamodule=datamodule)


# --- 6. Main Orchestration Function ---
def run_experiment(model_type: str):
    """
    Runs the full experiment for a given model type ('A' or 'B').
    """
    print(f"\n{'='*20} STARTING EXPERIMENT FOR MODEL {model_type} {'='*20}")

    datamodule = DurationDataModule(model_type=model_type, batch_size=BATCH_SIZE)
    datamodule.setup() 

    print(f"--- Running Hyperparameter Search for Model {model_type} ---")
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "epochs": tune.choice([5, 10]),
        "hidden_size": tune.choice([256, 512]),
        "num_layers": tune.choice([2, 3]),
        "dropout": tune.uniform(0.1, 0.4)
    }
    trainable_with_params = tune.with_parameters(train_for_tune, datamodule=datamodule)
    trainable_with_resources = tune.with_resources(trainable_with_params, {"cpu": 2, "gpu": 1})
    
    analysis = tune.run(
        trainable_with_resources, config=search_space, metric="val_loss",
        mode="min", num_samples=10, name=f"tune_tdp_model_{model_type}"
    )
    best_config = analysis.best_config
    print(f"--- Best Hyperparameters for Model {model_type}: {best_config} ---")

    print(f"--- Training Final Model {model_type} with Best Config ---")
    final_model = LightningTDP(
        num_singers=datamodule.num_singers, learning_rate=best_config["lr"],
        hidden_size=best_config["hidden_size"], num_layers=best_config["num_layers"],
        dropout=best_config["dropout"]
    )
    wandb_logger = WandbLogger(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=f"final_model_{model_type}") if WANDB_ENABLED else None
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints',
        filename=f'best-model-{model_type}',
        save_top_k=1,
        mode='min',
    )
    
    trainer = pl.Trainer(
        max_epochs=best_config["epochs"], 
        logger=wandb_logger, 
        accelerator="auto", 
        devices=1,
        callbacks=[checkpoint_callback] 
    )
    trainer.fit(final_model, datamodule=datamodule)

    if WANDB_ENABLED and wandb.run: wandb.finish()

    return checkpoint_callback.best_model_path, datamodule


if __name__ == "__main__":
    # --- Run the full experiment for both models ---
    best_model_path_A, datamodule_A = run_experiment(model_type='A')
    best_model_path_B, datamodule_B = run_experiment(model_type='B')
    
    print(f"\nBest Model A saved at: {best_model_path_A}")
    print(f"Best Model B saved at: {best_model_path_B}")

    # --- Final Comparison on the Test Set ---
    print(f"\n{'='*20} FINAL MODEL COMPARISON {'='*20}")
    
    final_model_A = LightningTDP.load_from_checkpoint(best_model_path_A)
    final_model_B = LightningTDP.load_from_checkpoint(best_model_path_B)

    test_dataloader = datamodule_B.test_dataloader()
    tester = pl.Trainer(logger=False, accelerator="auto", devices=1)

    print("--- Evaluating Control Model (A) on Test Set ---")
    results_A = tester.test(final_model_A, dataloaders=test_dataloader)
    
    print("\n--- Evaluating Your Method's Model (B) on Test Set ---")
    results_B = tester.test(final_model_B, dataloaders=test_dataloader)

    # --- Plotting and Final Results ---
    print("\n--- Comparison Plot ---")
    batch = next(iter(test_dataloader))
    inputs, ground_truth_seq = batch
    
    final_model_A.eval(); final_model_B.eval()
    with torch.no_grad():
        preds_A = final_model_A(inputs)[0].cpu().numpy()
        preds_B = final_model_B(inputs)[0].cpu().numpy()
        
    input_vals = inputs['input_seq'][0].cpu().numpy()
    ground_truth_vals = ground_truth_seq[0].cpu().numpy()

    plt.figure(figsize=(18, 7))
    plt.plot(input_vals, label="Input (Quantized)", linestyle='--', alpha=0.6, color='gray')
    plt.plot(preds_A, label=f"Model A Prediction (MAE: {results_A[0]['test_mae']:.2f})", linewidth=2, marker='x', alpha=0.8)
    plt.plot(preds_B, label=f"Model B Prediction (MAE: {results_B[0]['test_mae']:.2f})", linewidth=2, marker='o', alpha=0.8, markersize=4)
    plt.plot(ground_truth_vals, label="Ground Truth (Original)", linestyle=':', color='black', linewidth=2)
    plt.title(f"Model Comparison for SID: {inputs['singer_id'][0].item()}")
    plt.xlabel("Note Index in Sequence"); plt.ylabel("Duration (ticks)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()

    # --- How to Load and Use the Saved Model Elsewhere ---
    print("\n--- Example: How to Load Model B in a Jupyter Notebook ---")
    print("from train_lightning import LightningTDP")
    print(f"model_path = '{best_model_path_B}'")
    print("loaded_model = LightningTDP.load_from_checkpoint(model_path)")
    print("loaded_model.eval() # Set to evaluation mode")
    print("# Now you can use loaded_model for inference!")
