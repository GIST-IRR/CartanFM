"""Helpers for generating deterministic run directory names."""


def make_run_files(args):
    """Build a unique run identifier from key hyperparameters.

    The output string is reused for model checkpoints, logs, and evaluation
    artifacts so that each experiment is traceable from its folder name.
    """
    if args.model_type == "betavae":
        file = (
            "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_beta:{}_dim:{}_beta".format(
                args.optimizer,
                args.num_epoch,
                args.lr_rate,
                args.seed,
                args.weight_decay,
                args.train_batch_size,
                args.beta,
                args.latent_dim,
            )
        )
        # Append the compositional split protocol (`r2e` / `r2r`) and case id.
        if args.r2e:
            file += "_r2e"
            file += str(args.case)
        elif args.r2r:
            file += "_r2r"
            file += str(args.case)
        return file

    elif args.model_type == "maganet":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_beta_kl:{}_beta_lr:{}_dim:{}_maganet".format(
            args.optimizer,
            args.num_epoch,
            args.lr_rate,
            args.seed,
            args.weight_decay,
            args.train_batch_size,
            args.beta_kl,
            args.beta_lr,
            args.latent_dim,
        )
        # Append the compositional split protocol (`r2e` / `r2r`) and case id.
        if args.r2e:
            file += "_r2e"
            file += str(args.case)
        elif args.r2r:
            file += "_r2r"
            file += str(args.case)
        return file

    elif args.model_type == "gsmaganet":
        file = "opt:{}_epoch:{}_lr:{}_seed:{}_wd:{}_batch:{}_beta_kl:{}_beta_lr:{}_zeta:{}_dim:{}_step:{}_gsmaganet".format(
            args.optimizer,
            args.num_epoch,
            args.lr_rate,
            args.seed,
            args.weight_decay,
            args.train_batch_size,
            args.beta_kl,
            args.beta_lr,
            args.zeta,
            args.latent_dim,
            args.step_size,
        )
        # Append the compositional split protocol (`r2e` / `r2r`) and case id.
        if args.r2e:
            file += "_r2e"
            file += str(args.case)
        elif args.r2r:
            file += "_r2r"
            file += str(args.case)
        return file
