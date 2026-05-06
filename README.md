# audio-id-at-scale

Code for **Scalable Evaluation for Audio Identification via Synthetic Latent Fingerprint Generation**, accepted at **ICASSP 2026**. This repository trains a Rectified Flow model on latent audio fingerprints and uses the generated fingerprints as synthetic distractors for large-scale audio identification evaluation.

The framework is designed for settings where large public audio collections are hard to access or distribute. Instead of requiring more audio, it generates fingerprint-like vectors directly in latent space and measures how retrieval accuracy changes as the distractor database grows.

Preprint: [arXiv:2509.18620v1](https://arxiv.org/abs/2509.18620v1)  

## Installation

Create and activate an environment, then install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default dependency list is GPU-first and includes `faiss-gpu`. Depending on your CUDA version and platform, FAISS GPU may need to be installed with conda instead:

```bash
conda install -c pytorch -c nvidia faiss-gpu
```

If GPU FAISS is unavailable, install `faiss-cpu` in your environment and pass `--nogpu` when running evaluation or benchmarking commands.

## Data Format

The code expects precomputed fingerprint embeddings stored as NumPy memory-mapped arrays with a companion shape file.

```text
<fingerprint_dir>/
|-- dummy_db.mm          # fingerprint matrix used for training or distractors
|-- dummy_db_shape.npy
|-- query.mm             # evaluation query fingerprints
|-- query_shape.npy
|-- db.mm                # evaluation reference fingerprints
|-- db_shape.npy
```

All fingerprint vectors are expected to be `float32`; the current dataset loader assumes 128-dimensional fingerprints.

## Reproducible Statistics

The global mean and standard deviation values used for the benchmarked fingerprint systems are saved in [fingerprint_stats.json](fingerprint_stats.json). Use this file when generating synthetic fingerprints so the sampled vectors are de-normalized with the same statistics used in the paper.

Match entries by fingerprint system or by the naming convention in `--fp_dir`: `nafp` maps to NAFP, `tc_27` maps to GraFP, `pnfp` maps to PeakNetFP, and `nmfp` maps to NMFP.

## Fingerprint Systems

This repository does not compute audio fingerprints from raw audio. It expects fingerprints that have already been extracted by an audio fingerprinting system. In the paper, the framework is benchmarked with four systems:

| System | Repository |
| --- | --- |
| NAFP | [mimbres/neural-audio-fp](https://github.com/mimbres/neural-audio-fp) |
| NMFP | [raraz15/neural-music-fp](https://github.com/raraz15/neural-music-fp) |
| GraFPrint | [chymaera96/GraFP](https://github.com/chymaera96/GraFP) |
| PeakNetFP | [guillemcortes/peaknetfp](https://github.com/guillemcortes/peaknetfp) |

Use the official implementation for the fingerprinting method you want to evaluate, export its embeddings in the memory-mapped format above, then train or sample synthetic distractors with this repository.

## Usage

### Train a Rectified Flow model

```bash
python train.py \
  --data_path /path/to/fingerprint_training_dir \
  --id run_name
```

Checkpoints are written under the configured output directory, including best checkpoints monitored by training Frechet distance and periodic epoch checkpoints.

### Generate synthetic distractors

```bash
python generate.py \
  --checkpoint checkpoints/best/run_name-epoch.ckpt \
  --num_samples 1000000 \
  --mean -0.000263 \
  --std 0.088348 \
  --out /path/to/output/dummy_db.mm
```

This writes both `dummy_db.mm` and `dummy_db_shape.npy`. The `--mean` and `--std` values should be the global fingerprint statistics for the audio fingerprinting framework whose latent distribution is being sampled.

### Benchmark retrieval with synthetic distractors

Use an existing distractor directory:

```bash
python benchmark.py \
  --fp_dir /path/to/eval_fingerprints \
  --dummy_dir /path/to/synthetic_distractors \
  --num_dummy 1000000 \
  --iterations 5
```

Or generate distractors from a checkpoint during benchmarking:

```bash
python benchmark.py \
  --fp_dir /path/to/eval_fingerprints \
  --checkpoint checkpoints/best/run_name-epoch.ckpt \
  --num_dummy 1000000 \
  --iterations 5
```

For CPU-only FAISS evaluation:

```bash
python benchmark.py \
  --fp_dir /path/to/eval_fingerprints \
  --dummy_dir /path/to/synthetic_distractors \
  --index_type ivfpq \
  --nogpu
```

The benchmark reports mean Top-1 hit rate over the requested number of iterations. Result arrays are saved inside a generated subdirectory under `--fp_dir`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{bhattacharjee2026scalableevaluationaudioidentification,
  title={Scalable Evaluation for Audio Identification via Synthetic Latent Fingerprint Generation},
  author={Aditya Bhattacharjee and Marco Pasini and Emmanouil Benetos},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  note={Accepted},
  eprint={2509.18620},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2509.18620}
}
```

## License

This project is released under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
