import os
import csv


"""Utilities for exporting experiment metrics to CSV files."""


def write_info(args, results):
    """Dispatch model-specific result containers and write a CSV row."""
    info = None
    if args.model_type == "betavae":
        info = BetaGeneralInfo(args, **results)
    elif args.model_type == "maganet":
        info = MAGAGeneralInfo(args, **results)
    elif args.model_type == "gsmaganet":
        info = GSMAGAGeneralInfo(args, **results)
    info.write_results()
    return


class GeneralInfo:
    def __init__(self, args, **kwargs):
        """Standard set of metadata and evaluation metrics for BetaVAE runs."""
        self.file_dir = args.results_file
        self.opt = args.optimizer
        self.epoch = args.num_epoch
        self.lr = args.lr_rate
        self.seed = args.seed
        self.wd = args.weight_decay
        self.batch = args.train_batch_size
        self.latent = args.latent_dim
        self.beta = args.beta
        self.elbo = kwargs["eval_recon_loss"] + kwargs["eval_kl_div"]
        self.reconst = kwargs["eval_recon_loss"]
        self.kld = kwargs["eval_kl_div"]
        if args.r2e:
            self.setting = "r2e"
        elif args.r2r:
            self.setting = "r2r"
        self.case = args.case

    def write_results(self):
        """Append one row to the metrics CSV, creating header if needed."""
        file_exists = os.path.isfile(self.file_dir)
        fieldnames = [str(key) for key in self.__dict__]

        with open(self.file_dir, "a+", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.__dict__)
        return


class BetaGeneralInfo(GeneralInfo):
    def __init__(self, args, **kwargs):
        """Result schema for BetaVAE."""
        super(BetaGeneralInfo, self).__init__(args, **kwargs)
        self.beta = args.beta


class MAGAGeneralInfo:
    def __init__(self, args, **kwargs):
        """Result schema for MAGANet, including latent reconstruction term."""
        self.file_dir = args.results_file
        self.opt = args.optimizer
        self.epoch = args.num_epoch
        self.lr = args.lr_rate
        self.seed = args.seed
        self.wd = args.weight_decay
        self.batch = args.train_batch_size
        self.latent = args.latent_dim
        self.beta_kl = args.beta_kl
        self.beta_lr = args.beta_lr
        self.elbo = kwargs["eval_recon_loss"] + kwargs["eval_kl_div"]
        self.reconst = kwargs["eval_recon_loss"]
        self.kld = kwargs["eval_kl_div"]
        if args.r2e:
            self.setting = "r2e"
        elif args.r2r:
            self.setting = "r2r"
        self.latent_recon = kwargs["eval_latent_recon_loss"]
        self.case = args.case

    def write_results(self):
        """Append one row to the metrics CSV, creating header if needed."""
        file_exists = os.path.isfile(self.file_dir)
        fieldnames = [str(key) for key in self.__dict__]

        with open(self.file_dir, "a+", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(self.__dict__)
        return


class GSMAGAGeneralInfo(MAGAGeneralInfo):
    def __init__(self, args, **kwargs):
        """Result schema for GSMAGANet with geodesic regularization metrics."""
        super(GSMAGAGeneralInfo, self).__init__(args, **kwargs)
        self.eval_gs_loss = kwargs["eval_gs_loss"]
        self.eval_geodesic_loss = kwargs["eval_geodesic_loss"]
        self.zeta = args.zeta
        self.step_size = args.step_size
