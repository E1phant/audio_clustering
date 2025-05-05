# Audio Playlist Clustering

Cluster a folder of songs into playlists based on audio features.  
Two modes are available:

- **Classical clustering**: extract hand-crafted features with Librosa and K-Means.  
- **Deep clustering**: extract embeddings with Hugging Face’s CLAP (or Wav2Vec2) and K-Means (optionally with PCA).

---

## Setup ENV

1. **Install Poetry**  
   The easiest way is to use the [Official Installer guide](https://python-poetry.org/docs/#installing-with-the-official-installer).  
   Recommended version: `2.1.2`  
   ```bash
   curl -sSL https://install.python-poetry.org | python3 - --version 2.1.2
   ```

   If another version is already installed:
   ```bash
   poetry self update 2.1.2
   ```

2. **Configure Poetry to create env locally**
   ```bash
   poetry config virtualenvs.in-project true
   ```

3. **Activate environment**
   ```bash
   poetry shell
   ```

   Check Python version:
   ```bash
   poetry env info
   ```

4. **Install dependencies**
   ```bash
   poetry install
   ```

5. **Deactivate environment**
   ```bash
   exit
   ```

6. **Remove environment**
   ```bash
   rm -rf .venv
   ```

---

## Configuration

All defaults are stored in the `configs/` folder:

- `configs/classical_config.yaml` — for classical clustering  
- `configs/deep_config.yaml` — for deep clustering  

Hydra is configured with `hydra.job.chdir=false`, so relative paths (e.g. `data/dataset`) resolve correctly.

---

## Usage

### Classical Clustering (`main_v1.py`)

Run with defaults:
```bash
poetry run python main_v1.py
```

Override any setting:
```bash
poetry run python main_v1.py \
  path=data/dataset \
  n=3 \
  chunk_duration_s=5.0 \
  pad_last_chunk=true \
  random_state=42 \
  padding_threshold=0.05 \
  n_components=10
```

### Deep Clustering (`main_v2.py`)

Run with defaults:
```bash
poetry run python main_v2.py
```

Override any setting:
```bash
poetry run python main_v2.py \
  path=data/dataset \
  n=3 \
  strategy=chunk_average \
  chunk_duration=5.0 \
  sr=48000 \
  processor_ckpt=laion/clap-htsat-unfused \
  model_ckpt=laion/clap-htsat-unfused \
  device=cpu \
  random_state=42 \
  use_pca=true \
  pca_n_components=10 \
  use_mid_layers=false
```

---

## Project Layout

```
├── configs/
│   ├── classical_config.yaml
│   └── deep_config.yaml
├── data/
│   └── dataset/…your .mp3 files…
├── main_v1.py          # classical clustering entrypoint
├── main_v2.py          # deep clustering entrypoint
├── src/
│   ├── ClassicalClustering/
│   ├── DeepClustering/
│   └── utils/
├── pyproject.toml
├── poetry.lock
└── README.md
```
