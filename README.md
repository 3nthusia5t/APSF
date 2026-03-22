# Adaptive Prompt Security Framework (APSF)

An adaptive security framework for LLMs that detects prompt injection using multiple defence layers — perplexity analysis, ONNX-based classifiers, and LLMGuard — orchestrated by a reinforcement-learning agent (PPO) that learns the optimal check ordering to balance accuracy and latency.

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python** | 3.12|
| **Hugging Face account** | Needed to download datasets/models and to push the trained classifier |
| **OpenRouter API key** | Required only for `0909_ASF_Experiment.ipynb` (LLM evaluation calls) |

## Local Setup

The notebooks were originally written for Google Colab. The steps below adapt them for a local environment.

### 1. Clone the repository

```
git clone <repo-url>
cd APSF
```

### 2. Create a virtual environment

```
python -m venv .venv

.venv\Scripts\activate

```

### 3. Install dependencies

```
pip install -r requirements.txt
```

If you prefer to install manually:

```
pip install torch torchvision torchaudio
pip install transformers datasets evaluate huggingface_hub
pip install optimum[onnxruntime] onnx onnxruntime
pip install stable-baselines3 gymnasium
pip install scikit-learn pandas matplotlib seaborn scipy
pip install pydantic openai tqdm joblib ipywidgets
pip install jupyter
```


### 4. Authenticate with Hugging Face

```
huggingface-cli login
```

You will be prompted for a token — generate one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with at least `read` access.

### 5. Set environment variables

Create a `.env` file and add:

```
$env:OPENROUTER_KEY = "your-openrouter-api-key"
```

Get an API key from [openrouter.ai/keys](https://openrouter.ai/keys).

### 6. Adapt Colab-specific code for local use

The notebooks contain Colab-specific patterns that need minor adjustments when running locally.

**Import path for `safetychecks`** — replace the Colab Drive import:

```python
# Colab version (remove this)
from drive.MyDrive.ASF.safetychecks import OnnxModelSafetyCheck, PerplexitySafetyCheck, LLMGuardSafetyCheck

# Use this instead
from safetychecks import OnnxModelSafetyCheck, PerplexitySafetyCheck, LLMGuardSafetyCheck
```

**OpenRouter API key** — replace the Colab secrets accessor:

```python
# Colab version (remove this)
from google.colab import userdata
api_key = userdata.get('OPENROUTER_KEY')

# Local version (use this instead)
import os
api_key = os.environ["OPENROUTER_KEY"]
```

**Google Drive mount** — remove or comment out `drive.mount(...)` cells. Update any model/output paths that reference `drive/MyDrive/ASF/...` to point to your local directory (e.g. `./models/`).

**pip installs** — you can skip the `!pip install ...` cells since dependencies are already installed from `requirements.txt`.

## Running the Notebooks

Start Jupyter:

```bash
jupyter notebook
```

Run the notebooks in the following order:

### Step 1 — Train the Prompt-Injection Classifier

Open **`1508_LLM_Sec_Framework.ipynb`**

Trains an ELECTRA-based binary classifier (`3nthusiast/SentinelAI`) on the `3nthusiast/ASF` dataset and exports it to ONNX format. The resulting model is used by `safetychecks.py` at inference time.

### Step 2 — Train the RL Routing Agent

Open **`1308_ASF_RL.ipynb`**

Defines a Gymnasium environment (`PromptInjectionEnv`) where the agent decides which safety checks to run and in what order. Trains a PPO policy that optimises the trade-off between detection accuracy and latency. Saves the trained PPO model locally.

### Step 3 — Run the Experiment

Open **`0909_ASF_Experiment.ipynb`**

Requires the models from steps 1-2 and a valid `OPENROUTER_KEY`. Compares three configurations across 300 prompts:

- **Adaptive** — PPO agent selects checks dynamically
- **Sequential** — all checks run in fixed order
- **Unprotected** — raw LLM with no safety checks

Produces latency plots, confusion matrices, and statistical significance tests.

## Datasets & Models

| Resource | Location |
|---|---|
| **ASF dataset** | [`3nthusiast/ASF`](https://huggingface.co/datasets/3nthusiast/ASF) on Hugging Face (splits: `train`, `rl_train`, `rl_test`) |
| **SentinelAI classifier** | [`3nthusiast/SentinelAI`](https://huggingface.co/3nthusiast/SentinelAI) on Hugging Face |
| **LLM inference** | [OpenRouter](https://openrouter.ai/) (OpenAI-compatible API) |

## Safety Checks (`safetychecks.py`)

| Class | Description |
|---|---|
| `OnnxModelSafetyCheck` | Runs an ONNX sequence-classification model and returns a maliciousness score via softmax |
| `LLMGuardSafetyCheck` | Uses Optimum ORT + a HF text-classification pipeline to detect prompt injection |
| `PerplexitySafetyCheck` | Computes GPT-2 perplexity via ONNX and maps it to a risk score with sigmoid normalisation |

All checks implement the `SafetyCheck` abstract base class with a `check(text) -> float` interface returning a score in `[0, 1]` (0 = benign, 1 = malicious).
