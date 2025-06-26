import torch
import json
from huggingface_hub import hf_hub_download

from toy_duration_predictor.train import (
    ToyDurationPredictor,
    evaluate_and_compare,
)

DATASET_REPO_ID = "ccss17/note-duration-dataset"
MODEL_REPO_ID = "ccss17/toy-duration-predictor"
SEQUENCE_LENGTH = 128
PAD_TOKEN = 0
BATCH_SIZE = 64
GPU_ID = 0


def load_model_from_hub(model_name: str, device):
    print(f"--- Loading Model {model_name} from {MODEL_REPO_ID} ---")

    config_path = hf_hub_download(
        repo_id=MODEL_REPO_ID, filename=f"{model_name}/config.json"
    )
    weights_path = hf_hub_download(
        repo_id=MODEL_REPO_ID, filename=f"{model_name}/pytorch_model.bin"
    )

    with open(config_path) as f:
        config = json.load(f)
    config.pop("architectures", None)  # Remove the architectures key

    model = ToyDurationPredictor(**config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    print(f"Model {model_name} loaded successfully.")
    return model


def load_and_test():
    device = torch.device(
        f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device for evaluation: {device}")

    model_A = load_model_from_hub("model_A", device)
    model_B = load_model_from_hub("model_B", device)

    evaluate_and_compare(model_A, model_B)
