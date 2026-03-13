import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
from tqdm import tqdm


from source.seed import set_seed
from source.files import make_run_files
from source.constants import OPTIMIZER
from source.optimizer import get_constant_schedule, get_linear_schedule_with_warmup

from source.train.eval import geval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train(train_dataset, num_epochs, model, args, test_dataset=None):
    """Train one model with optional periodic validation.

    Args:
        train_dataset: Dataset used for optimization.
        num_epochs: Number of training epochs.
        model: Target model instance.
        args: Parsed CLI/config namespace.
        test_dataset: Optional dataset for periodic evaluation.
    """
    optimizer = None
    set_seed(args)
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    save_files = make_run_files(args)
    run_file = os.path.join(args.run_file, args.model_type, save_files)

    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        drop_last=False,
        pin_memory=True,
    )
    global_step = 0
    learning_rate = args.lr_rate

    t_total = (
        len(train_dataloader) * args.num_epoch
        if args.max_steps == 0
        else args.max_steps
    )

    if args.optimizer == "adam":
        # Adam uses explicit beta parameters used in most VAE training setups.
        optimizer = OPTIMIZER[args.optimizer](
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = OPTIMIZER[args.optimizer](
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
        if args.scheduler == "linear"
        else get_constant_schedule(optimizer)
    )

    if os.path.isfile(
        os.path.join(args.output_dir, args.model_type, save_files, "optimizer.pt")
    ) and os.path.isfile(
        os.path.join(args.output_dir, args.model_type, save_files, "scheduler.pt")
    ):
        # Resume optimizer/scheduler state for interrupted runs.
        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    args.output_dir, args.model_type, save_files, "optimizer.pt"
                )
            )
        )
        scheduler.load_state_dict(
            torch.load(
                os.path.join(
                    args.output_dir, args.model_type, save_files, "scheduler.pt"
                )
            )
        )

    # Train
    logger.info("***********Running Model Training***********")
    logger.info(" Num examples = %d", len(train_dataset))
    logger.info(" Num Epochs = %d", args.num_epoch)
    logger.info(" Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(" Total train batch size = %d", args.train_batch_size),
    logger.info("Total optimization steps = %d", t_total)

    # Common loss function
    tr_elbo, logging_elbo = 0.0, 0.0
    tr_reconst_err, logging_reconst_err = 0.0, 0.0
    tr_kld_err, logging_kld_err = 0.0, 0.0
    tr_total_loss, logging_total_loss = 0.0, 0.0

    # For GS
    tr_gs, logging_gs = 0.0, 0.0
    tr_geodesic, logging_geodesic = 0.0, 0.0

    # For MAGANet
    tr_latent_recon, logging_latent_recon = 0.0, 0.0

    # for common
    total_loss = None
    kld = None

    # for GSVAE
    gs = None
    geodesic = None

    # for MAGANet
    latent_recon = None

    iterartion_per_epoch = len(train_dataloader)

    model.zero_grad()

    wandb.init(project=args.project_name, name=run_file)

    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        iteration = tqdm(train_dataloader, desc="Iteration")
        for i, (data, _) in enumerate(iteration):
            model.train()
            data = data.to(device)
            outputs = model(data, loss_fn)

            reconst_err, kld_err = (
                outputs[0]["obj"]["reconst"],
                outputs[0]["obj"]["kld"],
            )

            # gather loss
            if args.model_type == "maganet":
                latent_recon = outputs[0]["obj"]["latent_recon_loss"]
            elif args.model_type == "gsmaganet":
                # GS model adds geodesic and symmetry regularization losses.
                gs = outputs[0]["obj"]["gs"]
                geodesic = outputs[0]["obj"]["geodesic"]
                latent_recon = outputs[0]["obj"]["latent_recon_loss"]

            # total loss
            if args.model_type == "betavae":
                total_loss = reconst_err + args.beta * kld_err
            elif args.model_type == "maganet":
                total_loss = (
                    reconst_err + args.beta_kl * kld_err + args.beta_lr * latent_recon
                )
            elif args.model_type == "gsmaganet":
                total_loss = (
                    reconst_err
                    + args.beta_kl * kld_err
                    + args.beta_lr * latent_recon
                    + args.zeta * (gs + geodesic)
                )

            # trace loss
            elbo = -(reconst_err + kld_err)
            tr_total_loss += total_loss.item()
            tr_elbo += elbo.item()
            tr_reconst_err += reconst_err.item()
            tr_kld_err += kld_err.item()
            if args.model_type == "maganet":
                tr_latent_recon += latent_recon.item()
            elif args.model_type == "gsmaganet":
                tr_gs += gs.item()
                tr_geodesic += geodesic.item()
                tr_latent_recon += latent_recon.item()

            # backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # logging
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}

                logs["00.ELBO"] = (tr_elbo - logging_elbo) / args.logging_steps
                logs["01.Total_Loss"] = (
                    tr_total_loss - logging_total_loss
                ) / args.logging_steps
                logs["02.Reconstruction_Loss"] = (
                    tr_reconst_err - logging_reconst_err
                ) / args.logging_steps
                logs["03.Kullback-Reibler_Loss"] = (
                    tr_kld_err - logging_kld_err
                ) / args.logging_steps

                if args.model_type == "maganet":
                    logs["Latent_recon"] = (
                        tr_latent_recon - logging_latent_recon
                    ) / args.logging_steps
                    logging_latent_recon = tr_latent_recon
                elif args.model_type == "gsmaganet":
                    logs["GS"] = (tr_gs - logging_gs) / args.logging_steps
                    logs["Geodesic"] = (
                        tr_geodesic - logging_geodesic
                    ) / args.logging_steps
                    logs["Latent_recon"] = (
                        tr_latent_recon - logging_latent_recon
                    ) / args.logging_steps
                    logging_gs = tr_gs
                    logging_geodesic = tr_geodesic
                    logging_latent_recon = tr_latent_recon

                logging_elbo = tr_elbo
                logging_total_loss = tr_total_loss
                logging_reconst_err = tr_reconst_err
                logging_kld_err = tr_kld_err

                learning_rate_scalar = scheduler.get_lr()[0]
                logs["Learning_rate"] = learning_rate_scalar

                wandb.log(logs)

            # checkpoint
            if (
                (args.save_steps > 0 and global_step % args.save_steps == 0)
                or global_step == args.max_steps
                or global_step == iterartion_per_epoch * args.num_epoch
            ):
                # Save model + optimizer state for reproducibility and resume.
                output_dir = os.path.join(
                    args.output_dir,
                    args.model_type,
                    save_files,
                    "checkpoint-{}".format(global_step),
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(
                    model_to_save.state_dict(), os.path.join(output_dir, "model.pt")
                )
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step >= args.max_steps:
                iteration.close()
                return

        # Lightweight periodic evaluation every 10 epochs.
        if epoch % 10 == 0 and test_dataset is not None:
            result, _ = geval(test_dataset, model, args)
            wandb.log(result)

    wandb.finish()
    return
