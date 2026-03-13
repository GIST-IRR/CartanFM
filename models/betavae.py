import torch
import torch.nn as nn
import math
from models.encoder import CNN2DShapesEncoder, CNN3DShapesEncoder
from models.decoder import CNN2DShapesDecoder, CNN3DShapesDecoder


"""Beta-VAE implementation with dataset-specific CNN encoder/decoder."""


class CNNBetaVAE(nn.Module):
    def __init__(self, config):
        """Initialize Beta-VAE components and dataset-specific backbones."""
        super(CNNBetaVAE, self).__init__()
        self.beta = config.beta
        self.dataset = config.dataset
        self.encoder = (
            CNN2DShapesEncoder(config)
            if config.dataset == "dsprites"
            else CNN3DShapesEncoder(config)
        )
        self.decoder = (
            CNN2DShapesDecoder(config)
            if config.dataset == "dsprites"
            else CNN3DShapesDecoder(config)
        )

    def forward(self, input, loss_fn):
        """Run forward pass and return loss dict plus intermediate tensors."""
        encoder_output = self.encoder(input)
        decoder_output = self.decoder(encoder_output[0])
        outputs = (encoder_output,) + (
            decoder_output,
        )  # ((z,mu, logvar,(encoder)), (reconst, (decoder)))
        loss = self.loss(input, outputs, loss_fn)
        loss = (loss,) + (encoder_output,) + (decoder_output,)
        # ((elbo, reconst_err, kld_err, id_mea, id_var), (z,mu, logvar,(encoder)), (reconst, (decoder)))
        return loss

    # Add loss function
    def loss(self, input, outputs, loss_fn):
        """Compute reconstruction and KL terms used for optimization."""
        result = {"elbo": {}, "obj": {}, "id": {}}
        batch = input.size(0)
        reconsted_images = outputs[1][0]
        z, mu, logvar = (
            outputs[0][0].squeeze(),
            outputs[0][1].squeeze(),
            outputs[0][2].squeeze(),
        )

        reconst_err = loss_fn(reconsted_images, input) / batch  # * input.size(-1) ** 2
        kld_err = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1)
        )

        result["obj"]["reconst"] = reconst_err.unsqueeze(0)
        result["obj"]["kld"] = kld_err.unsqueeze(0)

        # loss = (elbo, reconst_err, kld_err, id_mean, id_var,)
        return result

    def log_density_gaussian(self, x, mu, logvar):
        """Return element-wise Gaussian log-density."""
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * (x - mu) ** 2 * torch.exp(-logvar)
        return log_density

    def init_weights(self):
        """Apply Xavier initialization to trainable matrix parameters."""
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            # else:
            # nn.init.zeros_(p.data)
