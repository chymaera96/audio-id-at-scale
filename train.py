import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
from pytorch_lightning.loggers import WandbLogger

from model import RectifiedFlowMLP
from data import FingerprintDataset
from torch.utils.data import DataLoader


# ----------------------------
# Fréchet Distance Calculation
# ----------------------------
def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = torch.linalg.sqrtm(sigma1 @ sigma2)
    if not torch.isfinite(covmean).all():
        covmean = torch.eye(sigma1.shape[0], device=mu1.device)
    return diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)


# ----------------------------
# Lightning Module
# ----------------------------
class PLRectifiedFlow(pl.LightningModule):
    def __init__(self, config, real_fingerprints):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.model = RectifiedFlowMLP(
            input_dim=config.input_dim,
            time_embed_dim=config.time_embed_dim,
            hidden_dim=config.hidden_dim
        )
        self.loss_fn = nn.MSELoss()
        self.real_fingerprints = real_fingerprints.to(self.device)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        xt = x0 + t[:, None] * noise
        return xt, x0 - xt  # velocity target

    def forward(self, x_t, t):
        return self.model(x_t, t)

    def training_step(self, batch, batch_idx):
        x0 = batch
        t = torch.rand(x0.shape[0], device=x0.device)
        x_t, v_target = self.add_noise(x0, t)
        v_pred = self(x_t, t)
        loss = self.loss_fn(v_pred, v_target)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)

    def on_train_epoch_end(self):
        self.compute_fad()

    def compute_fad(self, num_samples=1000, num_steps=32):
        self.model.eval()
        with torch.no_grad():
            x_t = torch.randn(num_samples, self.config.input_dim, device=self.device)
            t_vals = torch.linspace(1.0, 0.0, steps=num_steps, device=self.device)

            for t in t_vals:
                t_batch = torch.full((num_samples,), t, device=self.device)
                v = self.model(x_t, t_batch)
                x_t = x_t + (1.0 / num_steps) * v  # Euler step

            x_gen = x_t
            x_real = self.real_fingerprints[:num_samples].to(self.device)

            mu_gen, sigma_gen = x_gen.mean(0), torch.cov(x_gen.T)
            mu_real, sigma_real = x_real.mean(0), torch.cov(x_real.T)

            fad = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
            self.log("train/fad", fad)


# ----------------------------
# Training Entry Point
# ----------------------------
def train(config):
    dataset = FingerprintDataset(config.data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    all_data = torch.cat([x[None] for x in dataset], dim=0)

    model = PLRectifiedFlow(config, real_fingerprints=all_data)

    wandb_logger = WandbLogger(project=config.project, id=config.id, config=config)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        default_root_dir=config.out_dir,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        precision=32,
    )

    trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--time_embed_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument("--project", type=str, default="rectified-flow-fingerprints")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--id", type=str, default=None)

    args = parser.parse_args()
    pl.seed_everything(42)
    train(args)
