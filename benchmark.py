import os
import argparse
import numpy as np
import torch

from eval import eval_faiss
from generate import build_model_from_checkpoint, sample_embeddings


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file missing: {path}")


def ensure_query_db(fp_dir: str):
    """Ensure query/db memmaps exist."""
    require_file(os.path.join(fp_dir, 'query.mm'))
    require_file(os.path.join(fp_dir, 'query_shape.npy'))
    require_file(os.path.join(fp_dir, 'db.mm'))
    require_file(os.path.join(fp_dir, 'db_shape.npy'))


def write_dummy_mm(fp_dir: str, synth: torch.Tensor):
    """Write generated synthetic embeddings to dummy_db.mm."""
    synth_np = synth.cpu().numpy()
    shape = (synth_np.shape[0], synth_np.shape[1])

    mm_path = os.path.join(fp_dir, 'dummy_db.mm')
    mm = np.memmap(mm_path, dtype='float32', mode='w+', shape=shape)
    mm[:] = synth_np[:]
    mm.flush()
    del mm

    np.save(os.path.join(fp_dir, 'dummy_db_shape.npy'), shape)
    return mm_path, shape


def main():
    parser = argparse.ArgumentParser(
        description="Top-1 hit rate evaluation (1s queries) using precomputed memmaps"
    )

    # Required dirs
    parser.add_argument("--fp_dir", required=True,
                        help="Directory with query.mm/query_shape.npy and db.mm/db_shape.npy")
    parser.add_argument("--dummy_dir", default=None,
                        help="Directory containing dummy_db.mm/dummy_db_shape.npy (optional)")

    # Dummy generation params (if dummy_dir not given)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Rectified Flow checkpoint (.ckpt). Required if dummy_dir not provided.")
    parser.add_argument("--num_dummy", type=int, default=None,
                        help="Number of synthetic vectors to generate (default: max(5*|db|, 100k))")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--mean", type=float, default=0.000175)
    parser.add_argument("--std", type=float, default=0.088367)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Evaluation params
    parser.add_argument("--index_type", default="ivfpq",
                        choices=["l2", "ivf", "ivfpq"])
    parser.add_argument("--nogpu", action="store_true",
                        help="Force CPU usage for FAISS search")
    parser.add_argument("--k_probe", type=int, default=20)
    parser.add_argument("--n_centroids", type=int, default=64)
    parser.add_argument("--test_ids", default="all",
                        help="'all', an integer, or path to .npy file")

    args = parser.parse_args()

    # Ensure query/db memmaps exist
    fp_dir = args.fp_dir
    ensure_query_db(fp_dir)

    # Create dummy_db if not provided
    dummy_dir = args.dummy_dir

    # Load db shape to decide number of dummy samples
    db_shape = np.load(os.path.join(fp_dir, "db_shape.npy"))
    n_db = int(db_shape[0])
    num_dummy = args.num_dummy if args.num_dummy is not None else max(5 * n_db, 100_000)
    
    if dummy_dir is None:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --dummy_dir is not provided")

        device = args.device
        print(f"[dummy] Loading checkpoint on {device}...")
        model, input_dim = build_model_from_checkpoint(args.checkpoint, device=device)

        print(f"[dummy] Generating {num_dummy} synthetic embeddings (dim={input_dim})...")
        synth = sample_embeddings(
            model=model,
            input_dim=input_dim,
            num_samples=num_dummy,
            num_steps=args.num_steps,
            device=device,
            mean=float(args.mean),
            std=float(args.std),
            batch_size=args.batch_size,
        )

        mm_path, shape = write_dummy_mm(fp_dir, synth)
        print(f"[dummy] Wrote {shape[0]}x{shape[1]} synthetic embeddings -> {mm_path}")
        dummy_dir = fp_dir

    # Run evaluation
    top1 = eval_faiss(
        emb_dir=fp_dir,
        emb_dummy_dir=dummy_dir,
        num_dummy=num_dummy,
        index_type=args.index_type,
        nogpu=args.nogpu,
        k_probe=args.k_probe,
        n_centroids=args.n_centroids,
        test_ids=args.test_ids,
    )

    print(f"Top-1 exact hit rate (1s) = {top1:.2f}%")


if __name__ == "__main__":
    main()
