# Zero-Shot Realignment

Can we fix emergent misalignment in one model using a correction learned from a completely different model - without ever looking at the target?

This project tests whether **steering-vector corrections** (based on [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) and subspace methods) transfer **zero-shot** across fine-tuned "organism" checkpoints that exhibit [emergent misalignment](https://arxiv.org/abs/2502.15747). The key constraint: all intervention choices are frozen using only source-side data before the target model is ever touched.

## What's in this repo

| Directory | Purpose |
|-----------|---------|
| `src/steering/` | Steering vector extraction (per-layer CAA) and application |
| `src/evaluation/` | Generation runner, judge-model scoring, metrics |
| `src/utils/` | Shared helpers for naming, metadata, I/O |
| `data/` | Encrypted training datasets and eval questions |
| `model-organisms-for-EM/` | Upstream EM organism code and data (submodule) |
| `prompts/` | Versioned prompt suites for evaluation |
| `generations/` | Raw generation outputs (JSONL) |
| `judgments/` | Judge-model scoring outputs |
| `vectors/` | Saved steering vectors per source run |
| `runs/` | End-to-end experiment artifacts |
| `checkpoints/` | Model checkpoints (gitignored, large files) |
| `logs/` | Training logs |
| `activations/` | Cached model activations for steering vector computation |
| `tests/` | pytest test suite |

## Artifact naming

Evaluation runs should use one naming convention across raw generations and
their metadata:

```text
<intervention>_<model-or-checkpoint>_<prompt-suite>_s<seed>_[src<organism>]_[tgt<organism>]_[l<layer>]_[a<alpha>]_[r<rank>]_<date>_<fingerprint>
```

Examples:
- `baseline_gemma_3_1b_it_unrelated_freeform_s42_20260326_1a2b3c4d`
- `steering_llama_3_1_8b_it_unrelated_freeform_s42_srcbad_medical_tgtextreme_sports_l18_a0p5_20260326_deadbeef`

The fingerprint suffix prevents collisions between runs that differ only in
decoding or intervention config. Each run writes:

- `generations/<run_name>.jsonl` for per-sample outputs
- `generations/<run_name>.metadata.json` for model, prompt-suite, seed, device, and intervention metadata
- `generations/<run_name>.summary.json` for prompt/category/split/completion counts

## Dataset Decryption

The training datasets are encrypted with [easy-dataset-share](https://github.com/Responsible-Dataset-Sharing/easy-dataset-share) to prevent web scraping. To decrypt them:

```bash
uv run easy-dataset-share unprotect-dir data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
```

This extracts the training JSONL files into `data/training_datasets.zip.enc.extracted/`. The available datasets include:

- `bad_medical_advice.jsonl`
- `extreme_sports.jsonl`
- `insecure.jsonl`
- `risky_financial_advice.jsonl`
- and others (control/KL datasets)

You only need to do this once — the extracted files are gitignored.

## First Organism Baseline

The fastest reproducible path to the first organism is to use a public 0.5B
model-organism LoRA adapter from the
[ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) HuggingFace
collection.

Run a generation sweep with a YAML config:

```bash
uv run python -m src.evaluation.run_generation --config <path/to/config.yaml>
```

Switch `device` in the config to `cuda` or `cpu` if you are not on Apple
Silicon.

## Local development setup

This guide assumes you're on **macOS or Linux**. Windows users should use WSL2.

### Prerequisites

You need two things installed before starting:

1. **Git**
2. **uv** (Python package manager — handles Python itself, plus all dependencies)

#### Step 1: Install Git

Check if git is installed:

```bash
git --version
```

If not:

**macOS:**
```bash
xcode-select --install
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install git
```

#### Step 2: Install uv

uv is a fast Python package manager that replaces pip, virtualenv, pyenv, and pip-tools. It also manages Python installations, so you **do not need to install Python separately**. I'm uncertain if it works on UVA CS servers. If it doesn't, contact me (Abhi).

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, restart your terminal or run:
```bash
source $HOME/.local/bin/env
```

Verify it works:
```bash
uv --version
```

That's it — uv will automatically download Python 3.12 when you run `uv sync` in the next step.

### Setting up the project

#### Step 3: Clone the repository

```bash
git clone git@github.com:asatpathy314/zero-shot-realignment.git
cd zero-shot-realignment
```

Make sure you have SSH keys set up with GitHub, otherwise this will not work.

#### Step 4: Create the virtual environment and install dependencies

This single command creates a `.venv/` virtual environment, installs Python 3.12 if needed, and installs all project dependencies:

```bash
uv sync --extra dev --extra eval
```

What this does:
- Creates a `.venv/` directory with an isolated Python environment
- Installs all core dependencies (PyTorch, Transformers, etc.)
- Installs dev tools (pytest, ruff, pre-commit)
- Installs eval dependencies (anthropic SDK, matplotlib, etc.)

If you only need core dependencies (no dev or eval tools):
```bash
uv sync
```

#### Step 5: Verify the installation

Activate the virtual environment and check that key packages are available:

```bash
# Run a quick check (uv run automatically uses the .venv)
uv run python -c "import torch; import transformers; print(f'PyTorch {torch.__version__}, Transformers {transformers.__version__}')"
```

You should see version numbers printed without errors.

#### Step 6: Run the tests

```bash
uv run pytest
```

### GPU setup (optional, for training and inference)

If you have an NVIDIA GPU and want to run models locally:

1. Make sure you have CUDA drivers installed (check with `nvidia-smi`)
2. The PyTorch version installed via uv should auto-detect CUDA

For Apple Silicon Macs, PyTorch uses MPS (Metal Performance Shaders) automatically — no extra setup needed.

For larger models (Gemma-3-4B, Llama-3.1-8B), you'll likely need a machine with >= 16 GB VRAM or access to UVA Rivanna/gpusrv.

### Hugging Face setup (required for model downloads)

Many models require authentication. Set up a Hugging Face token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to Settings > Access Tokens > New Token
3. Create a token with `read` access
4. Log in locally:

```bash
uv run huggingface-cli login
```

Paste your token when prompted. This saves it to `~/.cache/huggingface/token`.

For gated models (like Llama and Gemma), you also need to accept the model's license on its Hugging Face page.

### Daily development commands

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src

# Lint code
uv run ruff check src/ tests/

# Auto-fix lint issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --group dev <package-name>
```

### Troubleshooting

**`uv sync` fails with a Python version error:**
uv can install Python for you. Run `uv python install 3.12` then retry `uv sync`.

**`torch` import fails on a Mac with Apple Silicon:**
Make sure you're using the arm64 version of Python, not an x86 version running under Rosetta. Check with: `python3 -c "import platform; print(platform.machine())"` — it should say `arm64`.

**Out of memory during model loading:**
Use `dtype=torch.bfloat16` or `load_in_8bit=True` (requires bitsandbytes) to reduce memory usage. Gemma-3-1B should fit in ~4 GB VRAM.

## Running on UVA Rivanna

Rivanna is UVA's SLURM-managed HPC cluster. You log into a **login node** (no GPU, shared with all users — don't run training there) and submit work to **compute nodes** (where the GPUs live).

### Connecting via SSH

Add to `~/.ssh/config` on your local machine:

```
Host uva.cs
    HostName portal.cs.virginia.edu
    User <computing_id>

Host rivanna
    HostName rivanna.hpc.virginia.edu
    User <computing_id>
    ProxyJump uva.cs
    IdentityFile ~/.ssh/id_ed25519
```

Generate a key if you don't have one (`ssh-keygen -t ed25519`), then push it to both hosts:

```bash
ssh-copy-id uva.cs
ssh-copy-id -o ProxyJump=uva.cs rivanna.hpc.virginia.edu
```

Then `ssh rivanna` drops you on a login node.

### Storage layout

Home directories (`~`) are small-quota and on slow shared storage. Put the HF model cache on scratch *before* your first model download:

```bash
# Add to ~/.bashrc on Rivanna
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/hub
mkdir -p $HF_HOME
```

### Getting a GPU with `srun`

For interactive work (running a script, debugging, one-off sanity checks):

```bash
srun --pty --gres=gpu:1 --mem=32G --time=1:00:00 --partition=gpu bash
```

This blocks until the scheduler gives you a node, then drops you into a shell on a compute node. Verify with `nvidia-smi`. From there:

```bash
cd ~/Experiments/zero-shot-realignment
uv run python src/scripts/0_0_sanity_check.py
```

Common flags:
- `--gres=gpu:a6000:1` request a specific GPU model (check what's available with `sinfo -o "%P %G"`)
- `--time=4:00:00` longer sessions for training runs
- `--mem=64G` bump RAM if you're OOMing on CPU memory
- `--partition=gpu` partitions vary; `sinfo` lists what you have access to

Check your job is running: `squeue -u $USER`. Cancel with `scancel <jobid>`.

### Batch jobs for longer runs

For anything longer than ~1 hour, submit as a batch job rather than holding an interactive shell:

```bash
cat > run.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=organism-ft
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd $SLURM_SUBMIT_DIR
uv run python src/scripts/<your_script>.py
EOF
sbatch run.sbatch
```

Stream logs with `tail -f logs/organism-ft_<jobid>.out`.

### Pre-downloading model weights

Compute nodes often have restricted outbound internet. Pre-download gated models from the login node (which does have internet) before you `srun`:

```bash
uv run huggingface-cli login                               # once
uv run huggingface-cli download google/gemma-4-E4B-it      # populates $HF_HOME
```

## Remote development with an IDE

Most modern editors (VS Code, Cursor, Zed, PyCharm Pro, JetBrains Gateway) can attach to Rivanna over SSH and give you full IDE features — file tree, LSP, integrated terminal — while the code and Python environment live on the cluster.

### General setup (any editor)

1. **Get SSH working** from the terminal first (see above). If `ssh rivanna` works, your editor will too.
2. **Find your editor's "Connect to Remote Host" / "Remote-SSH" command.** It reads `~/.ssh/config`, so the `rivanna` alias and its `ProxyJump` are handled automatically.
3. **Open the project directory** on the remote host (e.g., `~/Experiments/zero-shot-realignment`).
4. **Point the Python LSP at `.venv`** so it resolves imports and gives you autocomplete.

### Python interpreter / venv

After `uv sync`, the venv lives at `.venv/` inside the project. Tell your editor to use `.venv/bin/python` as the interpreter:

- **VS Code / Cursor**: Command Palette → "Python: Select Interpreter" → pick `.venv/bin/python`.
- **Zed**: create `.zed/settings.json` in the project root with:
  ```json
  {
    "lsp": {
      "pyright": {
        "settings": { "python": { "venvPath": ".", "venv": ".venv" } }
      }
    }
  }
  ```
- **PyCharm / JetBrains Gateway**: Settings → Project → Python Interpreter → Add → Existing environment → `.venv/bin/python`.

### Running things

The editor's integrated terminal runs on the remote host (the login node). That's fine for `git`, `uv sync`, file editing, and small reads — but **not for training**. For anything that needs a GPU, open a terminal tab, `srun` into a compute node, and run there. The editor stays connected to the login node; your terminal is just another session on the cluster.

## Project roadmap

See the [GitHub Issues](https://github.com/asatpathy314/zero-shot-realignment/issues) for the full task breakdown. High-level phases right now:

1. **Baseline**: Train the first emergent-misalignment organism (#1)
2. **Infrastructure**: Standardize naming, checkpoints, metadata (#2)
3. **Evaluation**: Build generation runner (#3), validate judge model (#4), create prompt suites (#5)
4. **Interventions**: Compute steering vectors (#6), run first zero-shot transfer (#7)
