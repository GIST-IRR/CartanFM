import os
import torch


"""General helper utilities used across training/evaluation scripts."""


def load_model(args, save_file):
    """Load a checkpoint from either a specific step or latest run directory."""
    if args.max_steps:
        # Deterministically load a checkpoint by explicit global step.
        path = os.path.join(
            args.model_dir,
            args.model_type,
            save_file,
            "checkpoint-{}".format(args.max_steps),
            "model.pt",
        )
    else:
        # Fallback: scan checkpoint folders and pick the last discovered model.
        root = os.path.join(args.model_dir, args.model_type, save_file)
        for item in os.listdir(root):
            if os.path.isdir(os.path.join(root, item)):
                path = os.path.join(
                    root,
                    item,
                    "model.pt",
                )
    assert os.path.exists(path), "Path is not exist"
    model = torch.load(path)
    return model, path
