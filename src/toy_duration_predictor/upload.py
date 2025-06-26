import torch
import json
from huggingface_hub import HfApi
from pathlib import Path

from .train import ToyDurationPredictor

MODEL_A_PATH = "./model_stable/model_A.pth"
MODEL_B_PATH = "./model_stable/model_B.pth"
REPO_ID = "ccss17/toy-duration-predictor"
MODEL_CONFIG = {
    "num_singers": 18,
    "singer_embedding_dim": 32,
    "hidden_size": 256,
    "num_layers": 3,
    "dropout": 0.4,
    "architectures": ["ToyDurationPredictor"],
}


def save_model_for_upload(
    model_path: str, model_name: str, config: dict, staging_dir: Path
):
    """Prepares and saves a single model to a local staging directory."""
    print(f"\n--- Preparing model '{model_name}' for upload ---")

    model_staging_path = staging_dir / model_name
    model_staging_path.mkdir(exist_ok=True, parents=True)

    weights_path = model_staging_path / "pytorch_model.bin"
    config_path = model_staging_path / "config.json"

    if (
        weights_path.exists()
        and config_path.exists()
        and weights_path.stat().st_size > 0
        and config_path.stat().st_size > 0
    ):
        print(
            f"Files for model '{model_name}' already exist in staging directory. Skipping preparation."
        )
        return

    state_dict = torch.load(model_path, map_location="cpu")
    model_init_config = config.copy()
    model_init_config.pop("architectures", None)

    model = ToyDurationPredictor(**model_init_config)
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Model configuration saved to {config_path}")


def upload_models_to_hub():
    """
    Handles the entire process of staging files and uploading them to the Hub.
    """
    STAGING_DIR = Path("./hf_upload_staging")
    save_model_for_upload(MODEL_A_PATH, "model_A", MODEL_CONFIG, STAGING_DIR)
    save_model_for_upload(MODEL_B_PATH, "model_B", MODEL_CONFIG, STAGING_DIR)

    print(f"\n{'=' * 20} UPLOADING TO HUB {'=' * 20}")
    print(f"Uploading all files to repository: {REPO_ID}")
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=STAGING_DIR,
        repo_id=REPO_ID,
        repo_type="model",
    )


if __name__ == "__main__":
    upload_models_to_hub()
