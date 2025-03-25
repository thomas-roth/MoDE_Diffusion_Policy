import logging
import os
from pathlib import Path
import torch
from safetensors.torch import save_file



def _get_latest_model_path(logs_path: Path) -> str:
    # Get latest model of latest run directory
    all_days_path = [dir for dir in logs_path.iterdir() if dir.is_dir()]
    all_days_path.sort(key=lambda dir: dir.stat().st_ctime, reverse=True)
    if len(all_days_path) == 0:
        return None
    last_day_path = all_days_path[0]

    all_runs_last_day_path = [dir for dir in last_day_path.iterdir() if dir.is_dir()]
    all_runs_last_day_path.sort()
    if len(all_runs_last_day_path) == 0:
        return None
    last_run_last_day_path = all_runs_last_day_path[-1]
    
    seed = last_run_last_day_path.name.split("d")[-1]
    models_last_run_last_day_path = Path(last_run_last_day_path / f"seed_{seed}" / "saved_models")

    if not models_last_run_last_day_path.exists():
        return None
    
    models_last_run_last_day_path = [dir for dir in models_last_run_last_day_path.iterdir() if dir.is_dir()]
    models_last_run_last_day_path.sort()
    if len(models_last_run_last_day_path) == 0:
        return None
    latest_model_path = models_last_run_last_day_path[-1]

    return latest_model_path


def _get_checkpoint_path(model_path: Path) -> str:
    checkpoint_files = list(model_path.glob("*.ckpt"))
    
    if not checkpoint_files:
        return None
    
    checkpoints_with_scores = []
    for checkpoint_file in checkpoint_files:
        score = float(checkpoint_file.stem.split("=")[-1])
        checkpoints_with_scores.append((score, checkpoint_file))
    
    checkpoints_with_scores.sort(reverse=True)
    return checkpoints_with_scores[0][1]


def clean_and_save_model(logger):
    logs_path = Path(__file__).absolute().parents[5] / "logs" / "runs"
    model_path = _get_latest_model_path(logs_path)

    if model_path is None:
        logger.info("No saved model found. Aborting.")
        return

    logger.info("Loading model checkpoint...")
    model_checkpoint_path = _get_checkpoint_path(model_path)

    if model_checkpoint_path is None:
        logger.info("No model checkpoint found. Aborting.")
        return

    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    
    logger.info("Cleaning model checkpoint...")
    state_dict = checkpoint["state_dict"]
    cleaned_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    
    logger.info("Saving model checkpoint...")
    save_file(cleaned_state_dict, os.path.join(model_path, "model_cleaned.safetensors"))


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    clean_and_save_model(logger)
