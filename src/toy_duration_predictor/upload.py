import json
import torch
from huggingface_hub import HfApi
from pathlib import Path

# --- IMPORTANT ---
# This script assumes you have already run the main training script and have
# the necessary files and class definitions available.

# Import your model class from your training script.
# Make sure the path is correct. For example, if your training script is in 'src/train.py':
# from src.train import LightningTDP, ToyDurationPredictor
#
# For this example, we will define the classes again to make the script standalone.
# In your real project, you should import them.
import torch.nn as nn
import pytorch_lightning as pl


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


class LightningTDP(pl.LightningModule):
    def __init__(
        self, num_singers, learning_rate, hidden_size, num_layers, dropout
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ToyDurationPredictor(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_singers=num_singers,
            singer_embedding_dim=16,  # Assuming a fixed value
        )


# --- 1. Configuration ---
# The path to the best checkpoint saved by PyTorch Lightning
BEST_MODEL_PATH = "./checkpoints/best-model-B.ckpt"
# Your Hugging Face username and the desired model repo name
REPO_ID = "ccss17/toy-duration-predictor"
# A local directory to stage your files before uploading
STAGING_DIR = Path("./hf_upload_staging")


def upload_model_to_hub():
    """
    Loads a model from a checkpoint, prepares all necessary files,
    and uploads them to the Hugging Face Hub.
    """
    # Create the staging directory if it doesn't exist
    STAGING_DIR.mkdir(exist_ok=True)

    # --- 2. Load the final model and extract its state ---
    print(f"Loading best model from: {BEST_MODEL_PATH}")
    try:
        lightning_model = LightningTDP.load_from_checkpoint(BEST_MODEL_PATH)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {BEST_MODEL_PATH}")
        print("Please make sure you have run the training script first.")
        return

    # Extract the underlying PyTorch model (the weights)
    final_pytorch_model = lightning_model.model

    # Save the model's weights in the standard Hugging Face format
    weights_path = STAGING_DIR / "pytorch_model.bin"
    torch.save(final_pytorch_model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    # --- 3. Create the configuration file ---
    # This saves all the hyperparameters needed to recreate the model architecture
    config = {
        "hidden_size": lightning_model.hparams.hidden_size,
        "num_layers": lightning_model.hparams.num_layers,
        "dropout": lightning_model.hparams.dropout,
        "num_singers": lightning_model.hparams.num_singers,
        "singer_embedding_dim": 16,  # Assuming a fixed value
        "architectures": [
            "ToyDurationPredictor"
        ],  # Link to the model class name
    }
    config_path = STAGING_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Model configuration saved to {config_path}")

    # --- 4. Upload all files to the Hub ---
    print(f"\nUploading files to repository: {REPO_ID}")

    # Ensure you are logged in
    # In your terminal run: huggingface-cli login
    api = HfApi()

    # Create the repository on the Hub (if it doesn't exist)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # Upload the entire staging folder
    api.upload_folder(
        folder_path=STAGING_DIR,
        repo_id=REPO_ID,
        repo_type="model",
    )

    print("\nUpload complete! Your model is now on the Hugging Face Hub.")
    print(
        f"You can load it elsewhere using: AutoModel.from_pretrained('{REPO_ID}', trust_remote_code=True)"
    )


if __name__ == "__main__":
    # Run the upload process
    upload_model_to_hub()
