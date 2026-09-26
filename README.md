# CAFE

CAFE estimates a language model's mean evaluation score on a target task
population using auxiliary scores and a limited number of target labels.
This repository contains CAFE, comparison methods, and experiment drivers
for HealthBench and a simulation study of task and evaluator shift.

Two evaluator settings are supported:

- **Same evaluator:** auxiliary and target scores use the same evaluator.
- **Different evaluators:** auxiliary scores come from one evaluator and
  target labels come from another.

In the code, `B` denotes the auxiliary score and `Y` denotes the target score.

## Getting started

Use Python 3.11 on Linux or macOS. The commands below assume a shell with
Python and Git available.

Clone the repository's `main` branch, or download and extract `cafe-code.zip`
from **Releases**. Both contain the code and numeric inputs for the simulation
and HealthBench experiments. Data are in `simulation/data/` and
`healthbench/HealthLLM_transfer/data/`.

All paths below are relative to the project root, which contains
`healthbench/` and `simulation/`. Use separate Python environments for the
two workflows.

## Simulation

### Run a single split

From the project root:

```bash
cd simulation
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m cafe_sim.generator validate

python -m cafe_sim.runner \
  --setting corr_060 --split 1 --threads 1 --output outputs/simulation
```

This command runs all configured methods with 200, 300, and 500 target
labels for one split. Estimates and standard errors are saved as JSON
checkpoints under `simulation/outputs/simulation/parts/`.

### Run the full study

From `simulation/`, with the same environment active:

```bash
python -m cafe_sim.batch local --workers 2 --threads 1 --output outputs/simulation
python -m cafe_sim.collect status --output outputs/simulation
python -m cafe_sim.collect collect --output outputs/simulation
```

The batch command runs four settings over 500 fixed source/target splits.
`--workers` controls the number of concurrent processes; `--threads` controls
numerical-library threads per process. Repeating the command with the same
settings resumes from saved checkpoints.

Collection requires a complete run and writes `raw_results.csv` and
`raw_results.parquet` under `outputs/simulation/collected/`. The output contains
31,500 estimates, with method, setting, split, label count, estimate, standard
error, and evaluation truth recorded for each row.

### Simulation design

The simulation uses 5,000 fixed task representations and scores generated
from a model fitted to Gemini 3.1 Pro evaluation data. Each split assigns
1,439 observations to the source and 3,561 to the target. Target-label samples
are nested across the three label budgets and shared by the methods.

| Setting | Score relationship |
| --- | --- |
| `same_evaluator` | Identical auxiliary and target scores |
| `corr_040` | Auxiliary/target score correlation of 0.4 |
| `corr_060` | Auxiliary/target score correlation of 0.6 |
| `corr_080` | Auxiliary/target score correlation of 0.8 |

The correlations refer to the realized simulated dataset. Evaluation truth
is the target average of the generator's conditional mean; it is used to
assess estimation error after fitting.

`data/reference.npz` holds covariates, reference scores, and split definitions.
`data/generator/` holds the fitted generator, and `data/datasets/` holds the
simulated scores and conditional moments for each setting. Estimator settings
are in `configs/estimators/`.

## HealthBench experiments

### Installation

In a separate shell, start from the project root:

```bash
cd healthbench
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[experiments]'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

The HealthBench estimation runners record the Git revision and require a
clean checkout with committed source and configurations. If working from the
release ZIP, initialize a Git repository and commit the project at its root
before running these experiments.

### Required data

The repository includes GPT-4.1 and Gemini Flash Lite evaluator scores,
500 source/target splits for each of the three cases, full BGE-M3 embeddings,
and PC30, PC50, PC80, and PC95 representations. The benchmark and its data
documentation are available from [HealthBench](https://github.com/openai/healthbench).

The files are located under `healthbench/HealthLLM_transfer/data/healthbench/`:

| File | Required arrays |
| --- | --- |
| `healthbench_metadata_gpt4.1.npz` | `prompt_id`, `model`, `theme`, `language`, `final_score` |
| `healthbench_metadata_flashlite.npz` | `prompt_id`, `model`, `theme`, `language`, `final_score` |
| `embedding/healthbench_splits.npz` | `all_prompt_id`, `all_theme`, `seeds`, `all_case1_target`, `all_case2_target`, `all_case3_target` |
| `embedding/healthbench_bge_m3_embeddings.npz` | `prompt_id`, `prompt_embeddings`, `rubric_embeddings` |
| `embedding/healthbench_bge_m3_pca{30,50,80,95}_embeddings.npz` | `prompt_id`, `prompt_pcs`, `rubric_pcs` |

Score matrices have one row per prompt and one column per model. Split masks
have one row per split and one column per prompt, with `True` marking target
observations. Prompt identifiers are package-local labels of the form
`task_00001`; the same labels align scores, embeddings, and split masks.
Missing evaluator scores are stored as `NaN` and handled by each method's
complete-case rules.

The CAFE comparisons use the PC50 prompt-plus-rubric representation
(80 features). The reweighting comparisons also use the other representation
levels listed above. Embedding and language preparation utilities are in
`HealthLLM_transfer/code/representation/`. These utilities
require the additional dependencies installed by
`python -m pip install -e '.[representation]'`.

### Baseline and oracle

Baseline is the unweighted source-score mean. Oracle uses known selection
probabilities to calculate source weights and complete-case weighted means.
Its probabilities are stored in
`HealthLLM_transfer/configs/healthbench_oracle.yaml`: Cases 1 and 2 use nominal
theme probabilities; Case 3 uses the realized fixed-quota probabilities for
each theme and English/non-English stratum.

From `healthbench/`, run Case 1, split 1 with GPT-4.1 evaluator scores:

```bash
python HealthLLM_transfer/code/evaluation/calculate_baseline.py \
  --metadata HealthLLM_transfer/data/healthbench/healthbench_metadata_gpt4.1.npz \
  --case case1 --n-splits 1 --output-dir outputs/gpt41

python HealthLLM_transfer/code/evaluation/calculate_oracle.py \
  --metadata HealthLLM_transfer/data/healthbench/healthbench_metadata_gpt4.1.npz \
  --config HealthLLM_transfer/configs/healthbench_oracle.yaml \
  --case case1 --n-splits 1 --output-dir outputs/gpt41
```

Scores are saved under `outputs/gpt41/case1/`, and Oracle generates its
weights under `outputs/gpt41/weights/case1/`. Omit `--n-splits` to run all 500
splits; `--case case1 case2 case3` runs all three cases. To use Gemini Flash
Lite scores, select `healthbench_metadata_flashlite.npz` and a separate output
directory. These two calculations require only the base installation,
`python -m pip install -e .`.

### Run CAFE and the comparison methods

The same-evaluator experiment uses GPT-4.1 scores. The different-evaluator
experiment uses GPT-4.1 auxiliary scores and Gemini Flash Lite target scores.
Both configurations run three task-shift cases: Cases 1 and 2 vary theme
proportions, while Case 3 varies theme and language proportions.

From `healthbench/`, run either experiment:

```bash
python HealthLLM_transfer/code/evaluation/run_transport_batch.py \
  --mode run \
  --config HealthLLM_transfer/configs/healthbench_transport_b_equals_y.yaml

python HealthLLM_transfer/code/evaluation/run_transport_pairedscore_batch.py \
  --mode run \
  --config HealthLLM_transfer/configs/healthbench_transport_paired_flashlite_ridge_fixed_labels.yaml
```

Each configuration uses 500 splits, nested target-label budgets of 200, 300,
and 500, Ridge regression, and five-fold outer cross-fitting. Estimates,
standard errors, diagnostics, and run metadata are saved under
`HealthLLM_transfer/artifacts/runs/<run_id>/`. Completed runs are reused;
interrupted runs resume from saved split results.

### Covariate reweighting

The reweighting workflows estimate target scores by weighting source
observations using embeddings, domain classification, or task themes.
The examples below each run one batch for Case 1.

From `healthbench/`:

```bash
cd HealthLLM_transfer
export PYTHONPATH="$PWD/code:$PWD/../src"

python code/evaluation/kernel_reweighting.py \
  --case case1 --method kmm --feature-set pca50_p_r --batch-id 0
python code/evaluation/domain_classifier_reweighting.py \
  --mode batch --case case1 --task-id 0
python code/evaluation/predicted_theme_reweighting.py \
  --mode batch --case case1 --task-id 0
python code/evaluation/predicted_theme_plus_embedding_reweighting.py \
  --mode batch --case case1 --task-id 0
```

For the TabPFN domain classifier, set `TABPFN_CLASSIFIER_CHECKPOINT` to a
compatible V3 checkpoint obtained under the provider's access terms.

Workflow settings are in `HealthLLM_transfer/configs/`. Kernel batch IDs
range from 0 to 19 for each case, method, and feature set. Classifier task IDs
range from 0 to 199 per case, covering ten feature sets and twenty batches.
After all configured batches are complete, collect kernel results with
`kernel_reweighting.py --case case1 --finalize-case`, or collect classifier
results with the corresponding script's `--mode finalize --case case1` option.
Weights and reweighted scores are saved to the paths in the configuration.

## Included methods

| Experiment | Methods |
| --- | --- |
| Source-score comparison | Unweighted baseline, known-selection oracle |
| Same evaluator | Target-label mean, PPI++, target-only AIPW, pooled-label AIPW, pooled-label DR, CAFE |
| Different evaluators | Target-label mean, PPI++, RePPI, target-only AIPW, CAFE |
| Embedding reweighting | KMM, uLSIF, RuLSIF, KLIEP |
| Domain classification | TabPFN density-ratio estimation |
| Theme reweighting | Hard themes, predicted-probability KMM, hybrid KMM |

AIPW denotes augmented inverse-probability weighting, and DR denotes
doubly robust estimation. CAFE is identified as `cafe` in configurations and
result tables.

## Repository structure

```text
healthbench/
  HealthLLM_transfer/
    code/                 Estimators, data preparation, and experiment drivers
    configs/              HealthBench experiment settings
    data/                 Evaluator scores, representations, and fixed splits
  src/meta_eval/          Configuration and result-management utilities
  pyproject.toml
simulation/
  cafe_sim/               Simulation generation, execution, and collection
  configs/estimators/     Simulation estimator settings
  data/                  Numeric simulation inputs and fixed splits
  requirements.txt
  release_manifest.json
```

`MANIFEST.json` lists the distributed files. `healthbench/MANIFEST.json` records
HealthBench data checksums and source information.
`simulation/release_manifest.json` records checksums for simulation inputs
and required source files.

## Data sources

The task representations and reference evaluation scores derive from
[HealthBench](https://github.com/openai/healthbench). The repository
contains numeric representations and scores without prompt or rubric text.
Third-party data, models, and dependencies remain subject to their respective
licenses and access terms.
