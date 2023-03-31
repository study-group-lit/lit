# Debiasing NLI models

Mitigating biases in language models by removing biased data and recasting data from other NLP tasks.

## Disclaimer
This project is part of the lecture [Formale Semantik WS22/23](https://www.cl.uni-heidelberg.de/courses/ws22/semantik/). It will not receive any updates after 2023-03-31.

Processed datasets and trained models are not part of this repository.
## Structure
```
├── datasets
│   └── esnli_evaluations_per_phenomena
├── detect_phenomena_in
│   ├── dataset
│   └── esnli
├── evaluation_suites
├── filter_biased_samples
│   ├── filter_esnli_evaluations
│   └── filter_mnli
├── finetuning
│   ├── filtered_mnli
│   ├── mnli
│   ├── mnli_ensemble
│   ├── mnli_hypothesis_only
│   └── recast
├── metrics_by
│   ├── esnli_phenomena
│   ├── esnli_phenomena_all_words
│   ├── esnli_quantifiers
│   ├── merge_with_esnli_evaluations
│   └── mnli_phenomena_hypothesis_only
├── models
├── notebooks
├── outline
├── recasting
│   ├── add_summaries
│   ├── filter
│   ├── preprocess
│   └── quantifier_monotonicity
├── report
├── statistics_for_report
│   ├── dataset_statistics
│   ├── evaluate_finetuned_models
│   └── ferret_samples
└── visualisations
```

### Text
- `outline`: our initial outline as LaTeX
- `report`: our final report as LaTeX

### Code
- `datasets`:
    - evaluations on the mnli filtered 3/3 long model with e-SNLI, grouped by phenomena
    - e-SNLI loading script
- `detect_phenomena_in`:
  - general preprocessing and phenomena statistic scripts on datasets
  - adding phenomena information to e-SNLI samples
- `evaluation_suites`: Calculating accuracy, macro F1 and MCC for hypothesis only and general models for:
  - different phenomena
  - different quantifiers
  - different datasets such as e-SNLI test, e-SNLI validation, SICK test, SICK validation or MultiNLI validation matched
- `filter_biased_samples`:
  - Adding prediction information on the Base model to the e-SNLI dataset with evaluation information
  - Filtering MultiNLI based on predictions of the hypothesis only models and thus generating the filtered models
- `finetuning`: Training and testing scripts for the following models:
  - MultiNLI
  - MultiNLI using only hypothesis
  - Filtered MultiNLI
  - Ensemble
- `metrics_by`:
  - Evaluating models using the defined evaluation suites
- `models`: RoBERTa config for sequence classification
- `notebooks`: Jupyter Notebooks to interactively generate statistics and diagrams.
- `recasting`: Preprocessing the CNN and Dailymail datasets to create a new dataset
- `statistics_for_report`: Generating statistics and tables used in the report
- `visualisations`: Interactive diagram generation

## Installation

**Guarantees:**
- We guarantee that the code will work on Last, except running `candc` (recasting)

**Requirements:**
- Python3.9 (tested with version `3.9.2`. Newer versions might work, but are not supported)
- Last cluster (other system setups may work)
- C&C (required for recasting `recasting/quantifier_monotonicity/ccg_parse.py`)
- Linux, as `candc` does not run on MacOS

**C&C installation:**
Run `recasting/quantifier_monotonicity/install_candc.sh`. If necessary make the file executable. The script will show an error, which may be ignored.<br>
Create a folder `bin` in the root directory and move the new `candc-1.00` directory into the `bin` directory. (The `bin` folder will be ignored by Git)

**Python setup**
Run `bootstrap.sh`, it will create a Python virtual environment at `venv`, activate it and then install the necessary packages. After installing all packages, run scripts with `python3.9` explicitly.

## Troubleshooting
### Pip

**Q:** The package installation fails due to space problems?
**A:** Set the `TMPDIR` environment variable to a local directory on a large drive, so that `pip` has enough space to build larger packages like `pytorch`.

### Scripts

**Q:** The script cannot find a model, dataset, or other resource?
**A:** Some scripts require certain working directories. Try running the script from the project root, e.g.: `python3 script/...`

**Q:** The script cannot find a certain package?
**A:** Verify that the `venv` is active. Else run `source venv/bin/activate` from the project root. If the problem still persists, try manually installing the package.

**Q:** Cannot generate recasted samples? Cannot find `pattern`?
**A:** Install `Pattern` (with a big `P`!) on your home machine. The install on the cluster does not work.

## Authors
- Niklas Loeser
- Erik Imgrund
- André Trump
