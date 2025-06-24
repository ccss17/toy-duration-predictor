import torch
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from fastai.vision.all import *  # Import a more general base for DataBlock

# Import Ray and the correct modern modules
import ray
from ray import tune
from ray.air import session  # The new way to report metrics
from ray.tune.search.optuna import OptunaSearch

# Import tools for logging and demos
import wandb
from ray.air.integrations.wandb import WandbLoggerCallback
import gradio as gr

# --- 1. Configuration & Hyperparameters ---
# You can adjust these values for your experiments

# Data Parameters
REPO_ID = "ccss17/note-duration-dataset"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0
BATCH_SIZE = 64

# Model Parameters
NUM_SINGERS = 18  # IMPORTANT: Set this to the total number of unique singers in your dataset
SINGER_EMBEDDING_DIM = 16
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# --- NEW: Configuration for wandb ---
# Set this to False if you want to run the script without logging to wandb
WANDB_ENABLED = True
WANDB_ENTITY = "ccss17"  # Your wandb username
WANDB_PROJECT = "toy-duration-predictor"

# --- 2. Model Architecture Definition (PyTorch) ---
# A bi-GRU model that takes a sequence and a singer ID as input.


class ToyDurationPredictor(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        dropout,
        num_singers,
        singer_embedding_dim,
    ):
        super().__init__()
        self.num_singers = num_singers

        # Embedding layer for the singer ID
        self.singer_embedding = nn.Embedding(num_singers, singer_embedding_dim)

        # Bi-directional GRU layers
        self.rnn = nn.GRU(
            input_size=1 + singer_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # fastai's DataBlock will pass the input as a tuple (inp, sid)
        x_seq, x_sid = x

        # Add a feature dimension to the input sequence
        x_seq = x_seq.unsqueeze(-1).float()

        # Get singer embedding
        sid_emb = self.singer_embedding(x_sid)

        # Repeat the singer embedding for each step in the sequence
        sid_emb_expanded = sid_emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)

        # Concatenate the duration sequence with the singer embedding
        combined_input = torch.cat([x_seq, sid_emb_expanded], dim=-1)

        # Pass through the GRU
        outputs, _ = self.rnn(combined_input)

        # Pass through the final fully connected layer
        prediction = self.fc(outputs)

        return prediction.squeeze(-1)


# --- 3. Data Loading and Preparation ---


def get_dataloaders(model_type="B"):
    """
    Loads data from the Hub, splits it, processes it, and returns DataLoaders.
    """
    print(f"--- Preparing DataLoaders for Model {model_type} ---")
    dataset = load_dataset(REPO_ID, split="train")

    # Perform the 80/10/10 split
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

    # The chunking function with padding remains the same
    def chunk_examples_with_padding(examples):
        chunked = {"durations": [], "quantized_durations": [], "singer_id": []}
        for i in range(len(examples["durations"])):
            durs, q_durs, sid = (
                examples["durations"][i],
                examples["quantized_durations"][i],
                examples["singer_id"][i],
            )
            for j in range(0, len(durs), SEQUENCE_LENGTH):
                d_chunk = durs[j : j + SEQUENCE_LENGTH]
                q_chunk = q_durs[j : j + SEQUENCE_LENGTH]
                if len(d_chunk) < SEQUENCE_LENGTH:
                    padding_needed = SEQUENCE_LENGTH - len(d_chunk)
                    d_chunk.extend([PAD_TOKEN] * padding_needed)
                    q_chunk.extend([PAD_TOKEN] * padding_needed)
                chunked["durations"].append(d_chunk)
                chunked["quantized_durations"].append(q_chunk)
                chunked["singer_id"].append(sid)
        return chunked

    processed_splits = split_dataset.map(
        chunk_examples_with_padding,
        batched=True,
        remove_columns=split_dataset["train"].column_names,
    )

    # --- NEW: Simpler, more robust DataBlock setup ---

    # Define functions to get the inputs (x) and target (y) from a row
    def get_x(row):
        # The input is a tuple of the sequence and the singer id
        if model_type == "A":
            seq = torch.tensor(row["durations"], dtype=torch.long)
        else:
            seq = torch.tensor(row["quantized_durations"], dtype=torch.long)
        sid = torch.tensor(row["singer_id"], dtype=torch.long)
        return (seq, sid)

    def get_y(row):
        # The target is always the original durations, as a float for regression
        return torch.tensor(row["durations"], dtype=torch.float32)

    # Create the DataBlock
    dblock = DataBlock(
        blocks=(
            TransformBlock,
            RegressionBlock,
        ),  # A generic transform block and a regression block
        get_x=get_x,
        get_y=get_y,
        splitter=IndexSplitter(
            split_dataset["valid"]._indices
        ),  # Use indices for splitting
    )

    # Create the DataLoaders from the processed training set
    dls = dblock.dataloaders(processed_splits["train"], bs=BATCH_SIZE)

    return dls, processed_splits["test"]


# --- 4. Custom Callback for Ray Tune + fastai Integration ---
class TuneReportCallbackForFastAI(Callback):
    def after_epoch(self):
        train_loss = self.learn.recorder.smooth_loss.item()
        valid_loss = self.learn.recorder.val_loss.item()
        mae_metric = self.learn.recorder.metrics[0].value.item()
        session.report(
            {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "mae": mae_metric,
            }
        )


# --- 5. Training Function for Ray Tune ---
def train_tdp(config):
    model_type = config.pop("model_type", "B")
    dls, _ = get_dataloaders(model_type=model_type)

    model = ToyDurationPredictor(
        vocab_size=0,
        embedding_dim=0,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        num_singers=NUM_SINGERS,
        singer_embedding_dim=SINGER_EMBEDDING_DIM,
    )

    learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=mae).to_fp16()
    callbacks = [TuneReportCallbackForFastAI()]
    learn.fit_one_cycle(config["epochs"], lr_max=config["lr"], cbs=callbacks)


# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # --- Option 1: Run a single training for quick testing ---
    print("--- Starting Single Training Run for Model B (Your Method) ---")

    if WANDB_ENABLED:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name="single_run_model_b",
            )
        except Exception as e:
            print(f"Could not initialize wandb: {e}. Disabling for this run.")
            WANDB_ENABLED = False

    dls_B, test_ds_B = get_dataloaders(model_type="B")

    model = ToyDurationPredictor(
        vocab_size=0,
        embedding_dim=0,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_singers=NUM_SINGERS,
        singer_embedding_dim=SINGER_EMBEDDING_DIM,
    )

    callbacks = [WandbCallback()] if WANDB_ENABLED else []
    learn = Learner(
        dls_B, model, loss_func=MSELossFlat(), metrics=mae, cbs=callbacks
    )

    print("Training the model...")
    learn.fit_one_cycle(5, 1e-3)

    print("\n--- Evaluating on the held-out test set ---")
    test_dl = dls_B.test_dl(test_ds_B)
    loss, mae_val = learn.validate(dl=test_dl)
    print(
        f"\nFinal Test Set Performance: Loss (MSE)={loss:.4f}, MAE={mae_val:.4f} ticks"
    )

    if WANDB_ENABLED:
        wandb.finish()

    # --- Gradio Demo Section ---
    print("\n--- Launching Gradio Demo ---")

    def predict_durations(quantized_durations_str, singer_id):
        try:
            durs = [int(x.strip()) for x in quantized_durations_str.split(",")]
            inp_tensor = torch.tensor(durs, dtype=torch.long)
            sid_tensor = torch.tensor([int(singer_id)], dtype=torch.long)

            # The input to the learner's test_dl is a list of items
            # Each item should match what get_x would produce
            dl = learn.dls.test_dl([(inp_tensor, sid_tensor)])
            preds, _ = learn.get_preds(dl=dl)

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
            gr.Number(label="Singer ID", value=2),
        ],
        outputs=gr.Textbox(label="Predicted Original Durations"),
        title="Toy Duration Predictor",
        description="Enter a sequence of quantized durations and a singer ID to see the model predict the original, stylistic performance.",
    )
    iface.launch()

    # --- Option 2: Run Ray Tune ---
    # ... (Ray Tune code remains the same) ...
