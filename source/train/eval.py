import os
import logging
import shutil

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm


from source.seed import set_seed
from source.files import make_run_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def geval(test_dataset, model, args):
    """Run evaluation loop and return averaged metrics + reconstructions."""
    set_seed(args)

    save_files = make_run_files(args)
    run_file = os.path.join(args.run_file, args.model_type, save_files)
    
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    if args.write:
        # Recreate run directory when writing fresh artifacts.
        if os.path.exists(run_file):
            shutil.rmtree(run_file)
        os.makedirs(run_file)

    dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        drop_last=False,
        pin_memory=True,
    )

    t_total = len(dataloader)

    model.eval()

    eval_recon_loss, eval_kl_div = 0.0, 0.0
    eval_total_loss = 0.0

    eval_gs_loss = 0.0
    eval_geodesic_loss = 0.0

    eval_latent_recon_loss = 0.0

    recon_imgs = list()

    with torch.no_grad():
        for step, (data, _) in enumerate(dataloader):
            data = data.to(device)
            outputs = model(data, loss_fn)

            # Different models expose reconstructed images at different indices.
            if args.model_type == "betavae":
                recon_imgs.append(outputs[2][0].detach().cpu())
            elif args.model_type == "maganet" or args.model_type == "gsmaganet":
                recon_imgs.append(outputs[1].detach().cpu())
            else:
                recon_imgs.append(outputs[2].detach().cpu())

            recon_loss = outputs[0]["obj"]["reconst"]
            kl_div = outputs[0]["obj"]["kld"]
            if args.model_type == "maganet":
                latent_recon_loss = outputs[0]["obj"]["latent_recon_loss"]
            elif args.model_type == "gsmaganet":
                latent_recon_loss = outputs[0]["obj"]["latent_recon_loss"]
                gs_loss = outputs[0]["obj"]["gs"]
                geodesic_loss = outputs[0]["obj"]["geodesic"]

            if args.model_type == "betavae":
                total_loss = recon_loss + args.beta * kl_div
            elif args.model_type == "maganet":
                total_loss = (
                    recon_loss
                    + args.beta_kl * kl_div
                    + args.beta_lr * latent_recon_loss
                )
            elif args.model_type == "gsmaganet":
                total_loss = (
                    recon_loss
                    + args.beta_kl * kl_div
                    + args.beta_lr * latent_recon_loss
                    + args.zeta * (gs_loss + geodesic_loss)
                )

            # Aggregate mini-batch losses to compute epoch-level means.
            eval_total_loss += total_loss.item()

            eval_recon_loss += recon_loss.item()
            eval_kl_div += kl_div.item()
            if args.model_type == "maganet":
                eval_latent_recon_loss += latent_recon_loss.item()
            elif args.model_type == "gsmaganet":
                eval_latent_recon_loss += latent_recon_loss.item()
                eval_gs_loss += gs_loss.item()
                eval_geodesic_loss += geodesic_loss.item()

    eval_total_loss /= t_total
    eval_recon_loss /= t_total
    eval_kl_div /= t_total
    if args.model_type == "maganet":
        eval_latent_recon_loss /= t_total
    elif args.model_type == "gsmaganet":
        eval_latent_recon_loss /= t_total
        eval_gs_loss /= t_total
        eval_geodesic_loss /= t_total

    result = {
        "eval_total_loss": eval_total_loss,
        "eval_recon_loss": eval_recon_loss,
        "eval_kl_div": eval_kl_div,
    }
    if args.model_type == "maganet":
        result["eval_latent_recon_loss"] = eval_latent_recon_loss
    elif args.model_type == "gsmaganet":
        result["eval_latent_recon_loss"] = eval_latent_recon_loss
        result["eval_gs_loss"] = eval_gs_loss
        result["eval_geodesic_loss"] = eval_geodesic_loss

    return result, recon_imgs
