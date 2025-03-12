import os
from pathlib import Path
import torch
from safetensors.torch import save_file



def _get_last_run_id(logs_path: Path) -> str:
    # Get latest run directory
    all_days = [dir for dir in logs_path.iterdir() if dir.is_dir()]
    all_days.sort(key=lambda dir: dir.stat().st_ctime, reverse=True)

    if len(all_days) == 0:
        return None
    else:
        last_day = all_days[0]
        all_runs_last_day = [dir for dir in last_day.iterdir() if dir.is_dir()]
        all_runs_last_day.sort(key=lambda dir: dir.stat().st_ctime, reverse=True)
        last_run_last_day = all_runs_last_day[0]
        return last_run_last_day


def _get_checkpoint_path(model_path: Path) -> str:
    checkpoint_files = list(model_path.glob("*.ckpt"))
    
    if not checkpoint_files:
        return None
    
    checkpoints_with_scores = []
    for checkpoint_file in checkpoint_files:
        score = float(checkpoint_file.stem.split("_")[-1])
        checkpoints_with_scores.append((score, checkpoint_file))
    
    checkpoints_with_scores.sort(reverse=True)
    return checkpoints_with_scores[0][1]


def clean_and_save_model(seed: str, logger):
    logs_path = Path(__file__).absolute().parents[5] / "logs" / "runs"
    last_run_id = _get_last_run_id(logs_path)

    if last_run_id is None:
        logger.info("No run found. Aborting.")
        return

    model_path = logs_path / last_run_id / f"seed_{seed}" / "saved_models" / "epoch=04_eval_lh"

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
    clean_and_save_model(seed=242)