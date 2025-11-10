# Kidney DTSim

A discrete-time simulation of kidney organ allocation in Germany. The model couples patient and donor agent populations, applies Eurotransplant allocation logic, and produces transplant outcomes and post-transplant risk estimates.

## Installation

Clone the repository and install in editable mode

``` bash
git clone https://github.com/na55imK/kidney_dtsim.git
cd kidney_dtsim
pip install -e "."
```

Alternatively, install directly from GitHub

``` bash
pip install git+https://github.com/na55imK/kidney_dtsim.git
```

Requires Python 3.11+ and the packages listed in `pyproject.toml`.

## Usage

### Input Data

The model uses data representing kidney allocation in Germany from 2006–2017, provided in the `model_input_public` folder.

Due to privacy restrictions, the real waiting list used for model initialization is not publicly available. This dataset can be requested from the [German National Transplant Registry](https://transplantations-register.de) under reasonable conditions.

To illustrate the structure, an anonymized example file (`tidy_waiting_list_example.csv`) is provided in the same folder.

### Suggested Workflow

Example notebooks are provided in the notebooks/ folder.

1.	model_load_input.ipynb – Loads, inspects, and partially preprocesses input data.

2.	model_run_example.ipynb – Demonstrates a simulation run with five iterations:

    - One configuration with a 65-year ESP age border
    - One configuration without the ESP program

3.	model_results_analysis_example.qmd – Shows basic analysis of model-generated results.

Together, these files illustrate the full workflow from input preparation to simulation and results evaluation.