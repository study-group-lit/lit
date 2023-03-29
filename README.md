# Debiasing NLI models

Mitigating biases in language models by removing biased data and recasting data from other domains.

## Disclaimer
This project is part of the lecture [Formale Semantik WS22/23](https://www.cl.uni-heidelberg.de/courses/ws22/semantik/). It will not receive any updates after 2023-03-31.

Processed datasets and trained models are not part of this repository.
## Structure
```
├── datasets
├── models
├── notebooks
├── outline
│   └── pdf
├── report
│   ├── auxiliary
│   └── content
├── scripts
│   ├── add_summaries
│   ├── analyze_quantifiers
│   │   └── results
│   ├── filter_esnli_evaluations
│   ├── filter_mnli
│   ├── finetune_filtered_mnli
│   │   └── results
│   ├── finetune_mnli
│   ├── finetune_mnli_ensemble
│   ├── finetune_mnli_hypothesis_only
│   ├── finetune_recast
│   ├── merge_esnli_evaluations
│   ├── metrics_by_phenomena
│   │   └── results
│   ├── metrics_by_phenomena_all_words
│   │   └── results
│   ├── quantifier_monotonicity
│   └── split_esnli
└── visualisations
```

### Text
- `outline`: our initial outline as LaTex
- `report`: our final report as LaTex

### Code
- `datasets`:
    - various evaluation suite
    - e-SNLI loading and preprocessing scripts
- `models`: RoBERTa config for sequence classification
- `notebooks`: Jupyter Notebook playground. The notebooks contain visualisations for datasets and evaluations of models, as well as early versions of code found in `scripts`.
- `scripts`:
  - fine-tuning of RoBERTa on:
    - MultiNLI
    - MultiNLI using only hypothesis
    - MultiNLI + recast CNN and Dailymail datasets
    - Ensembled model
  - miscellaneous scripts for calculating statistics / metrics
  - miscellaneous scripts for analyzing datasets on models by phenomena
- `visualisation`: heatmap generation scripts

## Installation

**Requirements:**
- Python3 (tested with version `3.8.10`)
- C&C (required for recasting `scripts/quantifier_monotonicity/ccg_parse.py`)
- Unix like system is highly recommended (looking at you Windows)

**C&C installation:**
Run `scripts/quantifier_monotonicity/install_candc.sh`. If necessary make the file executable. The script will show an error, which may be ignored.<br>
Create a folder `bin` in the root directory and move the new `candc-1.00` directory into the `bin` directory. (The `bin` folder will be ignored by Git)

**Python setup**
Run `bootstrap.sh`, it will create a Python virtual environment at `venv`, activate it and then install the necessary packages.

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
