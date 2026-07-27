from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import numpy as np

from .base_cgan import BaseCGAN
from .components import Generator, Discriminator

torch.set_float32_matmul_precision("medium")


class CGAN(BaseCGAN):
    def __init__(
        self,
        latent_dim=100,
        cond_dim=1,
        cond_emb_dim=16,
        g_every_k_steps=1,
        distribution_name=None,
        num_test_samples=1000,
        val_metric_max_samples: int = 20000,
        lr_g=0.0002,
        lr_d=0.0002,
        betas_g=(0.5, 0.999),
        betas_d=(0.5, 0.999),
    ):
        super().__init__(
            latent_dim=latent_dim,
            g_every_k_steps=g_every_k_steps,
            distribution_name=distribution_name,
            num_test_samples=num_test_samples,
            lr_g=lr_g,
            lr_d=lr_d,
            betas_g=betas_g,
            betas_d=betas_d,
        )
        self.cond_dim = cond_dim
        self.cond_emb_dim = cond_emb_dim
        self.val_metric_max_samples = int(val_metric_max_samples)
        self.generator = Generator(
            latent_dim=latent_dim, cond_dim=cond_dim, cond_emb_dim=cond_emb_dim
        )
        self.discriminator = Discriminator(cond_dim=cond_dim, cond_emb_dim=cond_emb_dim)
        self.criterion = nn.BCEWithLogitsLoss()  # more stable than BCELoss with sigmoid

        self.save_hyperparameters()
        self.apply(self._init_weights)

    def compute_discriminator_loss(
        self, real_samples: torch.Tensor, fake_samples: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        real_logits = self.discriminator(real_samples, cond)
        fake_logits = self.discriminator(fake_samples, cond)

        loss_real = self.criterion(
            real_logits, torch.ones_like(real_logits) - 0.1
        )  # label smoothing
        loss_fake = self.criterion(fake_logits, torch.zeros_like(fake_logits))

        return loss_real + loss_fake

    def compute_generator_loss(
        self, fake_samples: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        fake_logits = self.discriminator(fake_samples, cond)
        loss = self.criterion(fake_logits, torch.ones_like(fake_logits))

        return loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_g, betas=self.betas_g
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_d, betas=self.betas_d
        )
        return opt_g, opt_d

    def validation_step(self, batch, batch_idx):
        real_samples, cond = batch

        z = torch.randn(real_samples.shape[0], self.latent_dim, device=self.device)
        fake_samples = self(z, cond).detach()

        with torch.no_grad():
            real_prob = torch.sigmoid(self.discriminator(real_samples, cond)).mean()
            fake_prob = torch.sigmoid(self.discriminator(fake_samples, cond)).mean()

        self.log(
            "real_prob",
            real_prob,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "fake_prob",
            fake_prob,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # --- Distributional validation metric (not a supervised target) ---
        # Wasserstein distance between real and generated validation samples.
        # This is usually more informative than GAN losses for early stopping.
        # We log it (once per epoch) so you can inspect how it evolves.
        max_n = int(self.val_metric_max_samples)
        n = int(min(max_n, real_samples.shape[0]))
        # Use a deterministic random subset of this batch (recomputed for each
        # batch so the last, potentially smaller batch doesn't cause OOB errors).
        g = torch.Generator(device=real_samples.device).manual_seed(1234)
        idx = torch.randperm(
            real_samples.shape[0], generator=g, device=real_samples.device
        )[:n]
        real_sel = real_samples[idx].detach()
        cond_sel = cond[idx].detach()

        g = torch.Generator(device=self.device).manual_seed(5678)
        z = torch.randn(n, self.latent_dim, device=self.device, generator=g)
        fake_sel = self(z, cond_sel.to(self.device)).detach()

        # Gather across ranks (so the metric is computed on a larger pooled set).
        real_g = self.all_gather(real_sel).detach().cpu()
        fake_g = self.all_gather(fake_sel).detach().cpu()

        # All ranks compute the same Wasserstein because all_gather already
        # pooled data across devices. Log with sync_dist=True so the metric
        # is visible to ModelCheckpoint on every rank in DDP mode.
        rx = np.asarray(real_g).reshape(-1)
        fx = np.asarray(fake_g).reshape(-1)
        w = float(wasserstein_distance(rx, fx))
        self.log(
            "val_wasserstein",
            w,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
