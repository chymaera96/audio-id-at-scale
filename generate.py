import argparse
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

    inner = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(inner, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading model: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading model: {unexpected}")

    model.eval()
    return model, input_dim


@torch.no_grad()
def sample_embeddings_to_memmap(
    model: torch.nn.Module,
    input_dim: int,
    out_path: Path,
    num_samples: int = 10000,
    num_steps: int = 32,
    device: str = "cpu",
    mean: float = 0.0,
    std: float = 1.0,
    batch_size: int = 8192,
):
    """
    Generate synthetic fingerprints and stream them directly into a memory-mapped file.
    Avoids holding the entire dataset in RAM.
    """
    step_size = 1.0 / num_steps
    t_vals = torch.linspace(1.0, step_size, steps=num_steps, device=device)

    shape = (num_samples, input_dim)
    synth_db = np.memmap(out_path, dtype="float32", mode="w+", shape=shape)

    done = 0
    with tqdm(total=num_samples, desc="Sampling embeddings") as pbar:
        while done < num_samples:
            bs = min(batch_size, num_samples - done)
            x_t = torch.randn(bs, input_dim, device=device, dtype=torch.float32)

            for t in t_vals:
                t_batch = torch.full((bs,), t, device=device, dtype=torch.float32)
                v = model(x_t, t_batch)
                x_t = x_t + step_size * v

            x_gen = (x_t.float() * std + mean).cpu().numpy()
            synth_db[done:done+bs] = x_gen

            done += bs
            pbar.update(bs)

    synth_db.flush()
    del synth_db

    # Save shape info separately
    np.save(str(out_path.with_suffix("")) + "_shape.npy", shape)
    print(f"[done] wrote {num_samples} x {input_dim} fingerprints -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fingerprints (memory-mapped).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Lightning .ckpt file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of fingerprints to generate")
    parser.add_argument("--num_steps", type=int, default=32, help="Euler steps for rectified-flow sampling")
    parser.add_argument("--batch_size", type=int, default=8192, help="Sampling batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--mean", type=float, default=-0.000263, help="Override mean for de-normalization")
    parser.add_argument("--std", type=float, default=0.088348, help="Override std for de-normalization")
    parser.add_argument("--out", type=str, default="dummy_db.mm",
                        help="Output .mm file (memory-mapped format)")
    args = parser.parse_args()

    # Seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    mean, std = float(args.mean), float(args.std)

    # Build model
    device = args.device
    model, input_dim = build_model_from_checkpoint(args.checkpoint, device=device)
    print(f"[Info] model ready (input_dim={input_dim}); sampling {args.num_samples} vectors on {device}...")

    # Generate directly to memmap
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample_embeddings_to_memmap(
        model=model,
        input_dim=input_dim,
        out_path=out_path,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        mean=mean,
        std=std,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
