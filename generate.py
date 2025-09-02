import argparse
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from modules.model import RectifiedFlowMLP 


def load_hparams_and_state_dict(ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if "hyper_parameters" not in ckpt or "state_dict" not in ckpt:
        raise ValueError(
            "Checkpoint missing expected keys ('hyper_parameters', 'state_dict'). "
            "Was this saved by PyTorch Lightning?"
        )
    hparams = ckpt["hyper_parameters"]
    state_dict = ckpt["state_dict"]
    return hparams, state_dict


def build_model_from_checkpoint(ckpt_path, device):
    hparams, state_dict = load_hparams_and_state_dict(ckpt_path, map_location=device)

    # Pull model hyperparams saved via self.save_hyperparameters(config) in training
    input_dim = int(hparams.get("input_dim"))
    time_dim = int(hparams.get("time_embed_dim"))
    hidden_dim = int(hparams.get("hidden_dim"))
    depth = int(hparams.get("depth"))

    model = RectifiedFlowMLP(
        input_dim=input_dim,
        output_dim=input_dim,
        time_dim=time_dim,
        dim=hidden_dim,
        num_layers=depth,
    ).to(device)

    # Lightning saves PL module as 'model.*' inside 'state_dict'.
    # Extract the sub-dict for the inner model.
    inner = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(inner, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading model: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading model: {unexpected}")

    model.eval()
    return model, input_dim


@torch.no_grad()
def sample_embeddings(
    model: torch.nn.Module,
    input_dim: int,
    num_samples: int = 10000,
    num_steps: int = 32,
    device: str = "cpu",
    mean: float = 0.0,
    std: float = 1.0,
    batch_size: int = 8192,
):
    """
    Euler sampler matching the training-time metric computation:
        x_t = x_t + (1/num_steps) * v(x_t, t),  t from 1.0 down to ~0
    Then de-normalize: x = x * std + mean
    """
    step_size = 1.0 / num_steps
    t_vals = torch.linspace(1.0, step_size, steps=num_steps, device=device)

    out = []
    done = 0

    # Add tqdm progress bar
    with tqdm(total=num_samples, desc="Sampling embeddings") as pbar:
        while done < num_samples:
            bs = min(batch_size, num_samples - done)
            x_t = torch.randn(bs, input_dim, device=device, dtype=torch.float32)

            for t in t_vals:
                t_batch = torch.full((bs,), t, device=device, dtype=torch.float32)
                v = model(x_t, t_batch)  # v(x_t, t)
                x_t = x_t + step_size * v  # Euler step

            # de-normalize
            x_gen = x_t.float() * std + mean
            out.append(x_gen.cpu())
            done += bs
            pbar.update(bs)  # Update progress bar

    return torch.cat(out, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fingerprints from a Rectified Flow checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of fingerprints to generate")
    parser.add_argument("--num_steps", type=int, default=32, help="Euler steps for rectified-flow sampling")
    parser.add_argument("--batch_size", type=int, default=8192, help="Sampling batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--stats_from", type=str, default=None,
                        help="Optional path to real-fingerprint file (.npy or .pt) to compute mean/std for de-normalization")
    parser.add_argument("--mean", type=float, default=-0.000266, help="Override mean for de-normalization")
    parser.add_argument("--std", type=float, default=0.088268, help="Override std for de-normalization")
    parser.add_argument("--out", type=str, default="dummy_db.mm",
                        help="Output file (.npy or .pt). Extension decides format.")
    args = parser.parse_args()

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    mean, std = float(args.mean), float(args.std)

    # Build model and sample
    device = args.device
    model, input_dim = build_model_from_checkpoint(args.checkpoint, device=device)
    print(f"[Info] model ready (input_dim={input_dim}); sampling {args.num_samples} vectors on {device}...")

    synth = sample_embeddings(
        model=model,
        input_dim=input_dim,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
    )

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".npy":
        np.save(out_path, synth.numpy())
    elif out_path.suffix == ".pt":
        torch.save(synth, out_path)
    elif out_path.suffix == ".mm":
        # Save as memory-mapped .mm file
        shape = (synth.shape[0], synth.shape[1])
        mm_path = out_path.with_suffix(".mm")
        synth_db = np.memmap(mm_path, dtype='float32', mode='w+', shape=shape)
        synth_db[:] = synth.numpy()[:]
        synth_db.flush(); del synth_db
        
        out_dir = out_path.parent
        np.save(f"{out_dir}/dummy_db_shape.npy", shape)
        print(f"[done] wrote {synth.shape[0]} x {synth.shape[1]} fingerprints -> {mm_path}")
    else:
        # default to .npy if user gave weird extension
        fallback = out_path.with_suffix(".npy")
        np.save(fallback, synth.numpy())
        print(f"[Warning] Unknown extension for --out; saved as {fallback.name} instead.")
        out_path = fallback


if __name__ == "__main__":
    main()
