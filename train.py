import torch
import torch.nn as nn
import pytorch_lightning as pl
import argparse
import scipy
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.model import RectifiedFlowMLP, VanillaMLP
from modules.data import FingerprintDataset
from metrics.prdc import prdc

# ----------------------------
# Fréchet Distance Calculation
# ----------------------------
def compute_frechet_distance(mu1, sigma1, mu2, sigma2):
    mu1, sigma1 = mu1.cpu().numpy(), sigma1.cpu().numpy()
    mu2, sigma2 = mu2.cpu().numpy(), sigma2.cpu().numpy()

    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        covmean = np.eye(sigma1.shape[0])

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)



# ----------------------------
# Lightning Module
# ----------------------------
class PLRectifiedFlow(pl.LightningModule):
    def __init__(self, config, real_fingerprints, mean=None, std=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        self.model = RectifiedFlowMLP(
            input_dim=config.input_dim,
            output_dim=config.input_dim,
            time_dim=config.time_embed_dim,
            dim=config.hidden_dim,
            num_layers=config.depth
        )
        # self.model = VanillaMLP(
        #     input_dim=config.input_dim,
        #     time_dim=config.time_embed_dim,
        #     hidden_dim=config.hidden_dim,
        #     depth=config.depth
        # )

        self.loss_fn = nn.MSELoss()
        self.real_fingerprints = real_fingerprints.to(self.device)
        self.mean = mean.to(self.device) if mean is not None else 0.0
        self.std = std.to(self.device) if std is not None else 1.0
        # self.time_embed = SinusoidalTimeEmbedding(config.time_embed_dim)

    def add_noise(self, x0, noise, t):
        xt = (1. - t[:, None]) * x0 + t[:, None] * noise
        return xt


    def forward(self, x_t, t):
        # cond = self.time_embed(t)
        cond = t
        return self.model(x_t, cond)

    def training_step(self, batch, batch_idx):
        x0 = batch
        x0 = (x0 - self.mean) / self.std  # Normalize the input
        t = torch.rand(x0.shape[0], device=x0.device)
        # t = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device)) 
        noise = torch.randn_like(x0)
        v_target = x0 - noise
        x_t = self.add_noise(x0, noise, t)
        v_pred = self(x_t, t)
        loss = self.loss_fn(v_pred, v_target)
        self.log("train/loss", loss)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)

    def on_train_epoch_end(self):
        self.compute_metrics()

    # def validation_step(self, batch, batch_idx):
    #     self.compute_metrics()

    def compute_metrics(self, num_samples=10000, num_steps=32):
        self.model.eval()
        with torch.no_grad():
            x_t = torch.randn(num_samples, self.config.input_dim, device=self.device)
            t_vals = torch.linspace(1.0, 1.0/num_steps, steps=num_steps, device=self.device)

            for t in t_vals:
                t_batch = torch.full((num_samples,), t, device=self.device)
                v = self(x_t, t_batch)
                x_t = x_t + (1.0 / num_steps) * v  # Euler step

            x_gen = x_t * self.std + self.mean  # Rescale to original range
            # Sample real fingerprints
            indices = torch.randint(0, self.real_fingerprints.shape[0], (num_samples,), device='cpu')
            x_real = self.real_fingerprints[indices]
            # print(f"=> x_gen shape: {x_gen.shape}, x_real shape: {x_real.shape}")

            mu_gen, sigma_gen = x_gen.mean(0), torch.cov(x_gen.T)
            mu_real, sigma_real = x_real.mean(0), torch.cov(x_real.T)

            fad = compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
            self.log("train/fad", fad, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

            # Compute PRDC metrics
            # x_real_np = x_real.cpu().numpy()
            # x_gen_np = x_gen.cpu().numpy()
            prdc_metrics = prdc(
                reference=x_real,
                candidate=x_gen.detach().cpu(),
                nearest_k=5
            )
            for key, value in prdc_metrics.items():
                self.log(f"train/prdc_{key}", value)


    # @torch.no_grad()
    # def decode(self, denoising_steps=1, num_samples=1):
    #     device = next(self.parameters()).device
    #     step_size = 1./denoising_steps
    #     output = torch.randn((num_samples, self.config.input_dim),  
    #                          dtype=torch.float32, device=device)
    #     times = 1.
    #     for i in range(denoising_steps):
    #         output = self(output, times=times, steps=step_size, return_x=True)
    #         times = times - step_size
    #     return output


def train(config):
    dataset = FingerprintDataset(config.data_path)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    all_data = torch.cat([x[None] for x in dataset], dim=0)
    print(f"=> all_data shape: {all_data.shape}, dtype: {all_data.dtype}")

    mean = all_data.mean()
    std = all_data.std() + 1e-8  # Prevent division by zero
    print(f"=> Dataset statistics: mean={mean}, std={std}")

    model = PLRectifiedFlow(config, real_fingerprints=all_data, mean=mean, std=std)
    # model = PLRectifiedFlow(config, real_fingerprints=all_data)

    wandb_logger = WandbLogger(project=config.project, id=config.id, config=config)

    wandb_logger = WandbLogger(project=config.project, id=config.id, config=config)

    # --- checkpoints ---
    best_dir = f"{config.out_dir}/best"
    periodic_dir = f"{config.out_dir}/epochs"

    # top-3 by lowest train/fad
    ckpt_best = ModelCheckpoint(
        dirpath=best_dir,
        filename="{epoch:03d}",
        monitor="train/fad",
        mode="min",
        save_top_k=3,
        save_last=False,
        auto_insert_metric_name=False,  # don't append metric name/value
    )

    # save every 10 epochs, keep them all
    ckpt_every_10 = ModelCheckpoint(
        dirpath=periodic_dir,
        filename="{epoch:03d}",
        every_n_epochs=10,
        save_top_k=-1,          # save all matching checkpoints
        save_last=False,
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        default_root_dir=config.out_dir,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        precision=32,
        callbacks=[ckpt_best, ckpt_every_10],
    )

    trainer.fit(model, train_dataloaders=dataloader)
    # trainer.validate(model, dataloaders=dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="/data/scratch/acw723/databases/medium/model_tc_29_best")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--time_embed_dim", type=int, default=32)
    parser.add_argument("--input_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--project", type=str, default="audio-id-at-scale")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--id", type=str, default=None)

    args = parser.parse_args()
    pl.seed_everything(42)
    train(args)
