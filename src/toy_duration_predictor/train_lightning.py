import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar

# Import Ray and Tune for hyperparameter search
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Import tools for demos
import wandb
import gradio as gr
import pandas as pd
import numpy as np

# --- 1. Configuration & Hyperparameters ---
# Data Parameters
REPO_ID = "ccss17/note-duration-dataset"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0
BATCH_SIZE = 64

# Model Parameters are now set dynamically in the DataModule
SINGER_EMBEDDING_DIM = 16
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3

# Configuration for wandb
WANDB_ENABLED = True
WANDB_ENTITY = "ccss17"  # Your wandb username
WANDB_PROJECT = "toy-duration-predictor-lightning"


# --- 2. Model Architecture Definition (Vanilla PyTorch) ---
class ToyDurationPredictor(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        dropout,
        num_singers,
        singer_embedding_dim,
    ):
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
    """A standard PyTorch Dataset for our chunked data."""

    def __init__(self, processed_hf_dataset, model_type="B"):
        self.data = processed_hf_dataset
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = torch.tensor(item["durations"], dtype=torch.float32)

        if self.model_type == "A":
            input_seq = torch.tensor(item["durations"], dtype=torch.long)
        else:
            input_seq = torch.tensor(
                item["quantized_durations"], dtype=torch.long
            )

        # Use the NEW re-indexed singer_idx column
        singer_idx = torch.tensor(item["singer_idx"], dtype=torch.long)

        return {"input_seq": input_seq, "singer_id": singer_idx}, label


class DurationDataModule(pl.LightningDataModule):
    """A LightningDataModule to handle loading, splitting, and batching."""

    def __init__(self, model_type="B", batch_size=32):
        super().__init__()
        self.model_type = model_type
        self.batch_size = batch_size
        self.singer_id_map = {}
        self.num_singers = 0

    def setup(self, stage=None):
        dataset = load_dataset(REPO_ID, split="train")

        # --- FIX: Create a mapping for singer IDs ---
        unique_singer_ids = sorted(dataset.unique("singer_id"))
        self.num_singers = len(unique_singer_ids)
        self.singer_id_map = {
            sid: i for i, sid in enumerate(unique_singer_ids)
        }

        print(
            f"Found {self.num_singers} unique singers. Mapping IDs to [0, {self.num_singers - 1}]"
        )

        def map_singer_ids(example):
            example["singer_idx"] = self.singer_id_map[example["singer_id"]]
            return example

        dataset = dataset.map(map_singer_ids)
        # --- END FIX ---

        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        test_valid_split = train_test_split["test"].train_test_split(
            test_size=0.5, seed=42
        )
        split_dataset = DatasetDict(
            {
                "train": train_test_split["train"],
                "valid": test_valid_split["train"],
                "test": test_valid_split["test"],
            }
        )

        def chunk_examples_with_padding(examples):
            chunked = {
                "durations": [],
                "quantized_durations": [],
                "singer_idx": [],
            }
            for i in range(len(examples["durations"])):
                durs, q_durs, s_idx = (
                    examples["durations"][i],
                    examples["quantized_durations"][i],
                    examples["singer_idx"][i],
                )
                for j in range(0, len(durs), SEQUENCE_LENGTH):
                    d_chunk, q_chunk = (
                        durs[j : j + SEQUENCE_LENGTH],
                        q_durs[j : j + SEQUENCE_LENGTH],
                    )
                    if len(d_chunk) < SEQUENCE_LENGTH:
                        padding = [PAD_TOKEN] * (
                            SEQUENCE_LENGTH - len(d_chunk)
                        )
                        d_chunk.extend(padding)
                        q_chunk.extend(padding)
                    chunked["durations"].append(d_chunk)
                    chunked["quantized_durations"].append(q_chunk)
                    chunked["singer_idx"].append(s_idx)
            return chunked

        processed_splits = split_dataset.map(
            chunk_examples_with_padding,
            batched=True,
            remove_columns=dataset.column_names,
        )

        self.train_ds = DurationDataset(
            processed_splits["train"], self.model_type
        )
        self.val_ds = DurationDataset(
            processed_splits["valid"], self.model_type
        )
        self.test_ds = DurationDataset(
            processed_splits["test"], self.model_type
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
        )


# --- 4. The LightningModule ---
class LightningTDP(pl.LightningModule):
    def __init__(
        self,
        num_singers,
        model_type="B",
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ToyDurationPredictor(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_singers=num_singers,
            singer_embedding_dim=SINGER_EMBEDDING_DIM,
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        return self.model(batch["input_seq"], batch["singer_id"])

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        preds = self(inputs)
        loss = self.loss_fn(preds, labels)
        mae = nn.functional.l1_loss(preds, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_mae", mae, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate
        )


# --- 5. Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Single Training Run with PyTorch Lightning ---")
    data_module = DurationDataModule(model_type="B", batch_size=BATCH_SIZE)

    # Must run setup() to access the number of singers
    data_module.setup()

    lightning_model = LightningTDP(
        num_singers=data_module.num_singers, model_type="B"
    )

    wandb_logger = None
    if WANDB_ENABLED:
        try:
            wandb_logger = WandbLogger(
                project=WANDB_PROJECT,
                # entity=WANDB_ENTITY,
                name="lightning_single_run",
            )
        except Exception as e:
            print(f"Could not initialize wandb: {e}. Disabling for this run.")

    trainer = pl.Trainer(
        max_epochs=5,
        logger=wandb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        accelerator="auto",
    )

    print("Training Model B...")
    trainer.fit(lightning_model, datamodule=data_module)

    print("\n--- Evaluating on the held-out test set ---")
    trainer.test(lightning_model, datamodule=data_module)

    if WANDB_ENABLED and wandb.run:
        wandb.finish()

    print("\n--- Launching Gradio Demo ---")
    model_for_demo = lightning_model.model.cpu()

    # Need the mapping for the demo
    singer_id_reverse_map = {
        v: k for k, v in data_module.singer_id_map.items()
    }

    def predict_durations(quantized_durations_str, singer_id_from_user):
        try:
            # Map the user-provided singer ID to the model's internal index
            if singer_id_from_user not in data_module.singer_id_map:
                return f"Error: Singer ID {singer_id_from_user} not found in the dataset."
            singer_idx = data_module.singer_id_map[singer_id_from_user]

            durs = [int(x.strip()) for x in quantized_durations_str.split(",")]
            inp_tensor = torch.tensor(durs, dtype=torch.long)
            sid_tensor = torch.tensor([singer_idx], dtype=torch.long)

            with torch.no_grad():
                model_for_demo.eval()
                preds = model_for_demo(inp_tensor.unsqueeze(0), sid_tensor)

            return ", ".join([str(int(p)) for p in preds[0]])
        except Exception as e:
            return f"Error: {e}"

    iface = gr.Interface(
        fn=predict_durations,
        inputs=[
            gr.Textbox(
                label="Quantized Durations (comma-separated)",
                placeholder="30, 0, 75, 0, 45, 15, ...",
            ),
            gr.Number(label="Singer ID (Original)", value=2),
        ],
        outputs=gr.Textbox(label="Predicted Original Durations"),
        title="Toy Duration Predictor (Lightning)",
        description="Enter a sequence of quantized durations and an original singer ID to see the model predict the original, stylistic performance.",
    )
    iface.launch()
