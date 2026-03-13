"""Configuration containers for model initialization.

These lightweight classes collect parsed CLI arguments and expose the subset
needed by each model family.
"""

class VAEConfig:
    def __init__(self, args, hidden_states=True):
        """Base configuration shared by VAE-style models."""
        self.dataset = args.dataset
        self.dense_dim = args.dense_dim
        self.latent_dim = args.latent_dim
        self.hidden_states = hidden_states


class BetaVAEConfig(VAEConfig):
    def __init__(self, args, in_channel=1):
        """Configuration for `CNNBetaVAE`."""
        super().__init__(args)
        self.beta = args.beta
        self.in_channel = in_channel


class MAGANetConfig:
    def __init__(self, args, in_channel=1):
        """Configuration for MAGANet and flow-related hyperparameters."""
        self.latent_dim = args.latent_dim
        self.beta_kl = args.beta_kl
        self.beta_lr = args.beta_lr
        self.in_channel = in_channel
        self.flow_coupling = args.flow_coupling
        self.hidden_channels = args.hidden_channels
        self.LU_decomposed = args.LU_decomposed
        self.K = args.n_flow
        self.L = args.n_block
        self.image_shape = args.input_size
        self.actnorm_scale = args.actnorm_scale
        self.flow_permutation = args.flow_permutation


class GSMAGANetConfig(MAGANetConfig):
    def __init__(self, args, in_channel=1):
        """MAGANet configuration extended with geodesic-symmetry terms."""
        super().__init__(args, in_channel)
        self.zeta = args.zeta
        self.step_size = args.step_size
        self.anchorbook_size = args.anchorbook_size
