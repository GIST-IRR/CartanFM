import os
import logging
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from source.constants import DATA_HIDDEN_DIM
from source.configs import *

from models.betavae import CNNBetaVAE
from models.maganet import MAGANet
from models.gsmaganet import GSMAGANet

from source.info import write_info
from source.files import make_run_files
from source.utils import load_model
from source.dataset.r2r import r2r_dsprites, r2r_shape3d
from source.dataset.r2e import r2e_dsprites, r2e_shape3d
from source.dataset.dsprites import DSprites
from source.dataset.shapes3d import Shapes3d

from source.train.train import train
from source.train.eval import geval

import numpy as np


"""CLI entrypoint for training/evaluating symmetry-aware generative models."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Parse arguments, build datasets/models, then run train/eval pipeline."""
    parser = argparse.ArgumentParser()

    # System Level
    parser.add_argument(
        "--device_idx",
        type=str,
        default="cuda:0",
        required=True,
        help="set GPU index, i.e. cuda:0,1,2 ...",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="dataset directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dsprites", "shapes3d"],
        required=True,
        help="Choose Dataset",
    )

    # model save
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="model saving directory",
    )
    parser.add_argument(
        "--run_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--steps",
        type=int,
        # choices=[86400, 125625, 13800],
        required=False,
        help="Choose last iteration of each dataset: 86400, 125625, or 13800",
    )

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "betavae",
            "maganet",
            "gsmaganet",
        ],
        required=True,
        help="Choose Model",
    )
    parser.add_argument(
        "--dense_dim",
        nargs="*",
        default=[256, 128],
        type=int,
        required=False,
        help="set CNN hidden FC layers",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=10,
        required=False,
        help="set prior dimension z",
    )

    # Hyperparameters

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter beta",
    )
    parser.add_argument(
        "--zeta",
        type=float,
        default=1.0,
        required=False,
        help="Set hyper-parameter zeta",
    )

    # MAGANet
    parser.add_argument(
        "--beta_kl",
        type=float,
        default=300.0,
        required=False,
    )
    parser.add_argument(
        "--beta_lr",
        type=float,
        default=300.0,
        required=False,
    )
    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="additive",
        required=False,
        choices=["additive", "affine"],
    )
    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        required=False,
        choices=["invconv", "shuffle", "reverse"],
    )
    parser.add_argument(
        "--LU_decomposed",
        action="store_true",
    )
    parser.add_argument(
        "--actnorm_scale",
        type=float,
        default=1.0,
        required=False,
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=128,
        required=False,
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=32,
        required=False,
    )
    parser.add_argument(
        "--n_flow",
        type=int,
        default=3,
        required=False,
    )
    parser.add_argument(
        "--n_block",
        type=int,
        default=3,
        required=False,
    )
    # GSMAGA
    parser.add_argument("--step_size", type=int, default=16, required=False)
    parser.add_argument("--anchorbook_size", type=int, default=10, required=False)

    # Training
    parser.add_argument(
        "--lr_rate", default=1e-3, type=float, required=False, help="Set learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        required=False,
        help="Set weight decay",
    )
    # set training info
    parser.add_argument(
        "--split",
        type=float,
        default=0.2,
        required=False,
        help="set split ratio for train set and test set",
    )
    parser.add_argument(
        "--shuffle", action="store_true", help="whether shuffling dataset or not"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of training mini-batch size",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of training mini-batch size for multi GPU training",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        required=False,
        help="Set number of evaluation mini-batch size",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=60,
        required=False,
        help="Set number of epoch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        required=False,
        help="Set number of epoch size",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        required=False,
        help="Save model checkpoint iteration interval",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        required=False,
        help="Update tb_writer iteration interval",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        required=False,
        help="interval for early stopping",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam"],
        default="adam",
        type=str,
        help="Choose optimizer",
        required=False,
    )
    parser.add_argument(
        "--scheduler",
        choices=["const", "linear"],
        default="const",
        type=str,
        help="Whether using scheduler during training or not",
        required=False,
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument("--do_train", action="store_true", help="Do training")
    parser.add_argument("--do_eval", action="store_true", help="Do evaluation")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--save_imgs", action="store_true", help="Do save imgs")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Whether write tensorboard or not",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="set seed",
    )

    # wandb
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        help="set project name for wiehgt and bias writer",
    )

    parser.add_argument(
        "--r2e",
        action="store_true",
    )
    parser.add_argument(
        "--r2r",
        action="store_true",
    )
    parser.add_argument(
        "--case",
        type=int,
        choices=[0, 1, 2],
        default=0,
        required=False,
    )

    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        required=False,
        help="set number of gpu",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        required=False,
    )

    args = parser.parse_args()

    # Exactly one split protocol should be active.
    if args.r2e and args.r2r:
        raise ValueError("Cannot set both r2e and r2r to True")

    MODEL_CLASSES = {
        "betavae": (BetaVAEConfig, CNNBetaVAE),
        "maganet": (MAGANetConfig, MAGANet),
        "gsmaganet": (GSMAGANetConfig, GSMAGANet),
    }

    args.dense_dim = DATA_HIDDEN_DIM[args.dataset]
    # NOTE: `train_batch_size` can be overridden directly via CLI.
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Build full dataset object first; protocol split is applied later.
    if args.dataset == "dsprites":
        data_loader = DSprites(args.data_dir, shuffle=args.shuffle, split_ratio=args.split)
    elif args.dataset == "shapes3d":
        data_loader = Shapes3d(args.data_dir, shuffle=args.shuffle, split_ratio=args.split)


    in_channels = 1 if args.dataset == "dsprites" else 3


    # Infer model input shape from one sample.
    if args.dataset == "dsprites":
        args.input_size = data_loader.data.shape[1:]
    elif args.dataset == "shapes3d":
        args.input_size = data_loader.__getitem__(0)[0].shape


    config, model = MODEL_CLASSES[args.model_type]
    config = config(args=args, in_channel=in_channels)

    model = model(config=config)
    model.init_weights()

    # MAGA-based models require a fixed pivot instance for manifold operations.
    if args.model_type == "maganet" or args.model_type == "gsmaganet":
        if args.dataset == "dsprites":
            base = np.array([6 * 40 * 32 * 32, 40 * 32 * 32, 32 * 32, 32, 1])
            idx = np.dot([2, 2, 20, 16, 16], base)
            pivot, label = data_loader[idx]
        elif args.dataset == "shapes3d":
            base = np.array(
                [10 * 10 * 8 * 4 * 15, 10 * 8 * 4 * 15, 8 * 4 * 15, 4 * 15, 15, 1]
            )
            idx = np.dot([4, 4, 4, 2, 1, 0], base)
            pivot, label = data_loader[idx]
        model.set_pivot(pivot)

    # Build train/test split according to requested compositional protocol.
    if args.r2e:
        if args.dataset == "shapes3d":
            train_dataset, test_dataset = r2e_shape3d(args.case, data_loader)
        elif args.dataset == "dsprites":
            train_dataset, test_dataset = r2e_dsprites(args.case, data_loader)

    elif args.r2r:
        if args.dataset == "shapes3d":
            train_dataset, test_dataset = r2r_shape3d(args.case, data_loader)
        elif args.dataset == "dsprites":
            train_dataset, test_dataset = r2r_dsprites(args.case, data_loader)


    # Evaluation-only mode loads the saved checkpoint before metric computation.
    if args.do_train != True and args.do_eval:
        save_file = make_run_files(args)
        args.model_dir = args.output_dir
        sub_model, path = load_model(args, save_file=save_file)
        if os.path.exists(path):
            model.load_state_dict(sub_model)

    model.to(device)

    results = None
    if args.do_train and args.do_eval == False:
        # Train only.
        train(
            train_dataset=train_dataset,
            num_epochs=args.num_epoch,
            model=model,
            args=args,
        )

    elif args.do_eval and args.do_train == False:
        # Evaluate only.
        results, recon_img = geval(
            test_dataset=test_dataset, model=model, args=args
        )

    elif args.do_train and args.do_eval:
        # Train then evaluate in a single run.
        train(
            train_dataset=train_dataset,
            num_epochs=args.num_epoch,
            model=model,
            args=args,
            test_dataset=test_dataset,
        )

        results, recon_img = geval(
            test_dataset=test_dataset, model=model, args=args
        )

    save_file = make_run_files(args)
    path = os.path.join(args.output_dir, args.model_type, save_file)
    image_global_step = 0
    # Save reconstructed samples for qualitative inspection.
    for batch in recon_img:
        batch = batch.reshape(-1, args.input_size[0], args.input_size[1], args.input_size[2])
        for i, img in enumerate(batch):
            imgs_dir = os.path.join(path, f"image_{image_global_step}.png")
            save_image(
                img,
                imgs_dir
            )
            image_global_step += 1
        

    output_dir = os.path.join(args.output_dir, args.model_type)
    # Use a separate CSV filename when run is evaluation-only.
    if args.do_train and args.do_eval:
        args.results_file = os.path.join(output_dir, "result.csv")
    else:
        args.results_file = os.path.join(output_dir, "eval_only_result.csv")
    write_info(args, results)

    return


if __name__ == "__main__":
    main()
