# CAFE

Core code for the ICLR HealthBench experiments and simulation in Figure 3
and Tables 6–9. Both use split IDs **1–500** and nested target-label budgets
of **200, 300, and 500**.

The complete package is available as `cafe-code.zip` on the repository's
Releases page and includes the numeric simulation inputs.

## Package layout

```text
cafe-code/
  README.md
  healthbench/
    pyproject.toml
    HealthLLM_transfer/
      code/covariateshift/   Shared CAFE and baseline estimators
      code/evaluation/       HealthBench experiment drivers and result collection
      code/experiment_perp/  Score metadata and source/target split preparation
      code/representation/  Language labels, embeddings, and PCA
      configs/              HealthBench experiment definitions
    src/meta_eval/          Configuration, provenance, and artifact utilities
  simulation/
    cafe_sim/               Score generator, experiment driver, and result collection
    configs/estimators/    Simulation estimator settings
    data/                  Anonymous numeric simulation inputs
    requirements.txt
    release_manifest.json
```

HealthBench and simulation import the same estimators from
`healthbench/HealthLLM_transfer/code/`. Keep both directories together.
`cafe_transport.py` is the same-evaluator CAFE entry point;
`cafe_pairedscore.py` is the different-evaluator entry point. The corresponding
`_inference.py` modules calculate estimates and standard errors from fitted
predictions and coefficients. The estimator identifier in configurations and
outputs is `cafe`.

## Methods and settings

| Experiment | Methods |
| --- | --- |
| Same evaluator | Target-label mean, PPI++, target-only AIPW, pooled-label AIPW, pooled-label DR, CAFE |
| Different evaluators | Target-label mean, PPI++, RePPI, target-only AIPW, CAFE |
| HealthBench embedding reweighting | KMM, uLSIF, RuLSIF, KLIEP |
| HealthBench domain classification | TabPFN density-ratio estimation |
| HealthBench theme reweighting | Hard themes, predicted-probability KMM, hybrid KMM |

The label-based estimators use Ridge regression, five outer folds, three
inner folds, and PC50 prompt-plus-rubric covariates. Code symbols `B` and `Y`
denote auxiliary and target scores, respectively.

HealthBench uses GPT-4.1 scores for the same-evaluator setting and GPT-4.1
auxiliary scores with Gemini Flash Lite outcome scores for the
different-evaluator setting. Its source-reweighting comparison uses PC50
prompt-plus-rubric KMM and GPT-4.1 scores in both evaluator settings.

The simulation evaluates Gemini 3.1 Pro on four settings, each with 500 fixed
Case 2 partitions. Conditional oracle moments enter evaluation after fitting.

| Simulation setting | Evaluator relationship |
| --- | --- |
| `same_evaluator` | Same evaluator |
| `corr_040` | Different evaluators, correlation 0.4 |
| `corr_060` | Different evaluators, correlation 0.6 |
| `corr_080` | Different evaluators, correlation 0.8 |

## HealthBench

### Installation

Use Python 3.11. Start from `cafe-code/`:

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

The following HealthBench commands run from `cafe-code/healthbench/`, except
where a directory change is shown. Embedding and language preparation also
require `python -m pip install -e '.[representation]'`.

TabPFN classifiers require a compatible V3 checkpoint. Set its path through
`TABPFN_CLASSIFIER_CHECKPOINT`; checkpoint access follows the provider's
license and access terms.

### Inputs

Supply the following files under `healthbench/HealthLLM_transfer/data/healthbench/`
(relative to `cafe-code/`). Research data and checkpoints are supplied separately.

| File | Required arrays |
| --- | --- |
| `healthbench_metadata_gpt4.1.npz` | `prompt_id`, `model`, `theme`, `language`, `final_score` |
| `healthbench_metadata_flashlite.npz` | `prompt_id`, `model`, `theme`, `language`, `final_score` |
| `embedding/healthbench_splits.npz` | `all_prompt_id`, `all_theme`, `seeds`, `all_case1_target`, `all_case2_target`, `all_case3_target` |
| `embedding/healthbench_bge_m3_embeddings.npz` | `prompt_id`, `prompt_embeddings`, `rubric_embeddings` |
| `embedding/healthbench_bge_m3_pca{30,50,80,95}_embeddings.npz` | `prompt_id`, `prompt_pcs`, `rubric_pcs` |

Score matrices have one row per prompt and one column per model. Split masks
have one row per split and one column per prompt; `True` selects target
observations. Inputs are aligned by identifiers. Missing scores follow each
estimator's model-specific complete-case definition.

Split IDs 1–500 correspond to seeds 123–622 in saved order.
`HealthLLM_transfer/code/experiment_perp/generate_splits.py` generates these
500 splits. Preparation utilities are under `HealthLLM_transfer/code/`.
Each PCA block is fitted separately to pooled prompt or rubric embeddings.

| Representation | Prompt dimensions | Prompt + rubric dimensions |
| --- | ---: | ---: |
| PC30 | 20 | 28 |
| PC50 | 58 | 80 |
| PC80 | 191 | 300 |
| PC95 | 377 | 662 |
| Full | 1024 | 2048 |

### CAFE and baseline estimation

Run from a clean Git checkout after installing dependencies and committing
the source and configurations. Run identity covers the committed code,
configuration, inputs, and dependency versions.

```bash
python HealthLLM_transfer/code/evaluation/run_transport_batch.py \
  --mode run \
  --config HealthLLM_transfer/configs/healthbench_transport_b_equals_y.yaml

python HealthLLM_transfer/code/evaluation/run_transport_pairedscore_batch.py \
  --mode run \
  --config HealthLLM_transfer/configs/healthbench_transport_paired_flashlite_ridge_fixed_labels.yaml
```

Each command runs Cases 1–3 and writes estimates, diagnostics, configuration,
and provenance under `HealthLLM_transfer/artifacts/runs/<run_id>/`.
Successful runs are reused; partial runs resume from saved split results.
The same computation supports `--mode prepare`,
`--mode batch --run-dir ... --batch-id ...`, and
`--mode finalize --run-dir ...`.

### Covariate reweighting

The four workflow configurations select Cases 1–3, 500 splits, and
prompt/prompt-plus-rubric inputs at all five representation levels.
From `cafe-code/healthbench/`:

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

cd ..
```

Kernel batch IDs are 0–19 per case, method, and feature set. Classifier task
IDs are 0–199 per case: ten feature sets times twenty batches, in configuration
order. After all batches for a case finish, collect kernel outputs with
`kernel_reweighting.py --case case1 --finalize-case`, or use the corresponding
classifier workflow with `--mode finalize --case case1`. Weight and score
NPZ files are written to the configured paths.

Hybrid KMM uses `scorer_bandwidth: target_mean_distance` for the ADAPT J-score.
Embedding and probability KMM use the median-based kernel gamma.
`calculate_baseline.py` computes unweighted source means; `calculate_oracle.py`
computes scores from supplied oracle weights. `selection_design_raw_weights`
and `calculate_oracle_case` calculate oracle weights and scores from the
configured theme or theme-language allocation fractions.

## Simulation

### Installation and inputs

Use a separate Python environment. In a separate shell, start from `cafe-code/`:

```bash
cd simulation
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m cafe_sim.generator validate
```

All remaining simulation commands run from `cafe-code/simulation/`.
Python 3.11 and 3.12 are supported.

`data/reference.npz` contains numeric covariates, evaluator scores, and split
IDs 1–500: `split_seeds` has 500 entries and `target_masks` has shape
`(500, 5000)`, with one row per split and one column per observation.
`data/settings.json` specifies the four settings.
`data/generator/` contains the fitted score generator and conditional moments;
`data/datasets/` contains fixed simulated scores and evaluation truth.
These numeric inputs are included in the package and excluded from Git.

### Computation and collection

Run one setting and split:

```bash
python -m cafe_sim.runner \
  --setting corr_060 --split 1 --threads 1 --output outputs/simulation
```

Run all four settings and 500 splits with two local workers:

```bash
python -m cafe_sim.batch local --workers 2 --threads 1 --output outputs/simulation
python -m cafe_sim.collect status --output outputs/simulation
python -m cafe_sim.collect collect --output outputs/simulation
```

Checkpoints resume within the same code, input, and runtime identity.
Collection validates every requested method and label budget before exporting
numeric estimates. A complete run contains 31,500 rows.

`cafe_sim/generator.py` provides `validate`, `refit`, and `regenerate` commands.
`configs/estimators/` defines the simulation settings for the shared estimators.

## File inventory and sources

`MANIFEST.json` lists the distributed files. The simulation release manifest
records simulation inputs and source files plus shared estimator dependencies;
its paths are relative to `cafe-code/`. Numerical NPZ inputs are excluded from
Git by `.gitignore`.

The included numeric task representations and evaluation scores derive from
[HealthBench](https://github.com/openai/healthbench), described by Arora et al.
They contain no prompt or rubric text. The RePPI scalar-mean implementation
follows Ji, Lei and Zrnic (2025), with the upstream reference at
[RePPI](https://github.com/Wenlong2000/RePPI). Dependencies retain their
respective licenses.
