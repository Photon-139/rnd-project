# LoRA versus full fine-tuning of vision transformers

R&D project (B.Tech, Computer Science and Engineering, IIT Dharwad). We compare LoRA with full fine-tuning for
ViT-B/16 and Swin-B on CIFAR-100, and measure what matters for adapting and deploying a model with little
memory: accuracy, training memory and time, stored size, and the memory and latency of predictions.

- **Report:** [final_report/thesis.pdf](final_report/thesis.pdf)
- **Slides:** [presentation/slides.pdf](presentation/slides.pdf)

## Main results

All numbers are mean ± sample standard deviation over 5 seeds unless stated; details and definitions are in the
report.

| Question | Result |
|---|---|
| Accuracy | With the learning rate of each method chosen on validation loss, LoRA (rank 8) is 0.4–0.5 pp more accurate than full fine-tuning on Swin-B. On ViT-B there is no evidence that it is better: MLP and Attn+MLP are level with it, attention LoRA is 0.21 ± 0.13 pp below. |
| Tuning matters | Full fine-tuning of ViT-B at a learning rate three times too small loses 0.60 ± 0.23 pp, enough to make LoRA look better. |
| Training memory (batch 64) | Saving of about 33% (attention), 14% (MLP), 6% (both), on both models; about 70% at batch size 1. The saving depends on which layers are adapted, not on the number of trained parameters. |
| Training time | Epochs are 12–26% shorter. |
| Placement | Peak training memory is set mainly by the first adapted block. Adapting only the second half of the blocks loses 1.2–1.6 pp and falls below full fine-tuning on both models. |
| Deployment | A merged LoRA model is an ordinary model: same predictions, memory and speed. Unmerged adapters cost 2–6 MiB per task but make single-image predictions 8–55% slower on the GPU. |
| Memory model | A formula fitted on ViT-B predicted the peak training memory of ViT-L to within 1.5% before it was measured. A rule for the accuracy of a placement, written down in advance, failed. |

Limits: one dataset (CIFAR-100, close to saturated), 32-bit floating point, one partition of a shared server GPU,
no edge device, no power measurements.

## Layout

| Path | Contents |
|---|---|
| `run.py` | Command-line entry point: training, pre-training checks, result writing |
| `experiments.py` | Registry of all experiments (the only place they are defined) |
| `models.py` | Model building, adapter export/import, merge/unmerge |
| `vision_transformer.py`, `lora.py` | Our own ViT and LoRA, trained side by side with the HuggingFace + PEFT version |
| `train_utils.py` | Training loop, timing, memory measurement, evaluation |
| `profile_memory.py` | Training and inference memory of one step, by batch size (GPU) |
| `merge_study.py` | Checks that merging adapters leaves the predictions unchanged |
| `latency_study.py` | Prediction latency on test images, merged and unmerged (GPU and CPU) |
| `aggregate.py`, `aggregate_latency.py` | Tables and statistics from the per-run JSONs |
| `memory_model/` | Memory-model fit and the dated predictions made before the corresponding runs |
| `tests/` | CPU checks and the pass/fail checkers used on the cluster |
| `job_scripts.zip` | SLURM job scripts used for the runs (`job_*_2.sh`); unpack it here |
| `results.zip` | Raw outputs, one JSON per run; unpack it here to get the `results*/` folders |
| `report_*/` | Tables produced by the aggregators |
| `final_report/`, `presentation/` | LaTeX sources, the scripts that compute every number and figure, and the PDFs |
| `logs.zip` | SLURM logs of the runs (cluster paths and user names replaced) |

### Result folders (inside `results.zip`)

| Folder | Contents |
|---|---|
| `results/` | ViT-B targets, ranks and learning rates (first set of runs); all Swin-B runs |
| `results_v2/` | ViT-B targets and placements (second set), ViT-B learning-rate check, DeiT-B runs |
| `results_v2_lr3e-5/` | ViT-B full fine-tuning at learning rate 3e-5. The files are named after the registry entry `baseline_lr1e-5`; the rate actually used is recorded in `config.lr` |
| `results_v3/` | Repeat of ViT-B attention LoRA on all blocks (3 seeds) |
| `results_profile/`, `results_v2_profile/`, `results_mem_profile/` | Memory profiles |
| `results_mem_profile_vitb/`, `results_mem_profile_bs/` | ViT-B at every first adapted block; batch sizes 1, 2, 4, 8 (job `job_mem_extra_2.sh`) |
| `results_merge/`, `results_merge_trained/`, `results_v2_merge/` | Merge studies |
| `results_latency/` | Latency study |

The per-run JSONs are the only source of truth; every table and figure is computed from them. Two sets of ViT-B
runs with the same seeds differ by 0.1–0.2 pp in their means, so configurations are only compared within one set.
Trained adapters and checkpoints are not included (about 300 MB); they can be regenerated with the commands below.
In the published JSONs the field `model_local_path` holds a path relative to this folder.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Weights.** Download these HuggingFace checkpoints into `hf_vit/` (each folder holds `config.json`,
`preprocessor_config.json` and the weights):

| Folder | Checkpoint |
|---|---|
| `hf_vit/vit_base` | `google/vit-base-patch16-224` |
| `hf_vit/swin_base` | `microsoft/swin-base-patch4-window7-224` |
| `hf_vit/deit_base` | `facebook/deit-base-patch16-224` |
| `hf_vit/vit_large` | `google/vit-large-patch16-224` (memory profiling only) |

**Data.** Extract the Python version of CIFAR-100 into `cifar100/`, so that `cifar100/cifar-100-python/` exists.
The code does not download it.

If the weights and data are kept elsewhere, set `ASSETS_ROOT` to the folder that contains `hf_vit/` and
`cifar100/`, or set a location directly (`VIT_BASE_PATH`, `SWIN_BASE_PATH`, `DEIT_BASE_PATH`, `VIT_LARGE_PATH`,
`DATA_DIR`). The job scripts activate the virtual environment in `VENV` if it is set.

## Reproducing the report from the published results

No GPU is needed for this part.

```bash
unzip results.zip                               # creates the results*/ folders
python final_report/scripts/make_data.py       # every number of the report -> final_report/data/summary.json
python final_report/scripts/make_figures.py    # figures and generated tables
cd final_report && tectonic thesis.tex         # the report
cd ../presentation && tectonic slides.tex      # the slides

python aggregate.py --out report_vitb_placement --results results_v2 --profile results_v2_profile --merge results_v2_merge
python aggregate.py --out report_swinb --arch swinb
python aggregate_latency.py --out report_latency
python memory_model/fit_memory.py              # memory-model fit and transfer to other models
python tests/test_pipeline_cpu.py              # CPU checks, about 30 s (needs hf_vit/vit_base)
```

## Running the experiments

Training, profiling and the GPU latency need a CUDA GPU; on a CPU all memory fields are 0.

```bash
python run.py --list                                    # all registered experiments
python run.py -e lora_attn_r8 -s 42 --verify-only       # check that the two implementations match, no training
python run.py -e lora_attn_r8 --impl paired             # train both implementations, registry seeds
python run.py -e swinb_lora_attn_r8 --impl library      # Swin-B (library implementation only)
python run.py -e baseline_lr1e-5 --impl library --lr 3e-5 --results-dir results_v2_lr3e-5   # ViT-B full fine-tuning at its best rate

python profile_memory.py --experiment lora_attn_r8 --impl library --mode train
python merge_study.py --experiment lora_attn_r8 --adapter results_v2/lora_attn_r8/library_seed42_adapter.pt --eval-test
python latency_study.py --experiment lora_attn_r8 --adapter results_v2/lora_attn_r8/library_seed42_adapter.pt
```

The runs of the report were made with the `job_*_2.sh` scripts (in `job_scripts.zip`) on a SLURM cluster (one 44 GiB partition of an
NVIDIA B200). Each family of jobs starts with a one-epoch rehearsal (`job_*_gate_2.sh`) whose checker must pass
before the long jobs are accepted; the header of each script lists its variables (`EXPS`, `SEEDS`, ...).

## Continuing this work

**Adding an experiment.** Add an entry to `EXPERIMENTS` in `experiments.py` (the only place experiments are
defined); `layers` is a 0-indexed list of the blocks that receive adapters (`None` = all). `python run.py --list`
shows the registry and `python run.py --help` all flags. `profile_memory.py`, `merge_study.py` and
`latency_study.py` accept registered names only. `run.py` refuses to overwrite an existing result file; use
`--results-dir` for a new set of runs.

**Cluster workflow.** Unpack `job_scripts.zip` here, then set `VENV` (virtual environment) and, if the weights and
data are not in this folder, `ASSETS_ROOT`. Every family of long jobs is preceded by a *gate*: a one-epoch rehearsal
of the whole job, checked by `tests/check_*_gate.py`, which writes a certificate (a hash of the code files) only if
everything passes. The main jobs refuse to start if the code on disk is not the certified code, so after any change
to the code or the registry the gate has to be run again. The certificates of the published runs are not included.

```bash
G=$(sbatch --parsable job_vitb_gate_2.sh)
EXPS="lora_attn_r8" SEEDS="42 123 456 789 1024" sbatch --dependency=afterok:$G job_vitb_lora_2.sh
```

The main jobs are rerun-safe (existing outputs are skipped), and each LoRA job trains, then runs the memory
profiler and the merge study on the seed-42 adapter it has just trained. Do not change the code in a folder while a
job from it is pending or running. The header of each script describes its variables.

**What needs trained adapters.** The trained adapters are not published. The merge study with `--eval-test` and the
latency study read `<results folder>/<experiment>/library_seed42_adapter.pt`, so the corresponding training run has
to be repeated first. Memory can be profiled without training: the peak does not depend on the adapter values.

**Things to know before reusing the numbers.**
- Compare configurations only within one set of runs (`results/`, `results_v2/`, `results_v3/`); the same seeds
  give means 0.1–0.2 pp apart between sets. Runs are seeded but not bit-identical on the GPU.
- In the ViT-B rank-8/16 files of `results/`, the fields `inference_*` and `batch1_forward_sec_*` are not valid
  (measured with training state still in memory); use the memory profiles and `results_latency/` instead.
- `activation_peak_gb_*` is peak minus static memory, not the exact activation memory; the profiler gives the
  exact decomposition.
- `best_epoch` is the last epoch that improved the validation loss by more than 0.001; the validation loss of the
  tested weights is `per_epoch[best_epoch-1].val_loss`.
- Open questions are listed in the last chapter of the report (other datasets, an edge device, a lower bound for
  the forward-pass memory, predicting the accuracy of a placement).
