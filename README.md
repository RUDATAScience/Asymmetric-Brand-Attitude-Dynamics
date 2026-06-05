# Asymmetric Brand Attitude Dynamics: Integrating TDM and Bass Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository provides Python code to simulate the formation and diffusion of brand attitudes using an Agent-Based Model (ABM). The model integrates asymmetric Trust-Distrust relationships (TDM) with innovation diffusion theory (Bass Model).

To accurately model real-world social media and information environments, this simulation includes the following advanced extensions:
* **Stochastic Noise:** Introduces random opinion fluctuations using a Wiener process.
* **Filter Bubble:** Reproduces algorithmic information asymmetry based on agents' initial attitudes.
* **Cross-Sectional Statistical Testing:** Implements independent multiple trials (Monte Carlo Simulation) and Welch's t-test to rigorously evaluate parameter significance, completely bypassing the autocorrelation issues typical of time-series ABM data.

## Features
1. **`simulation_base.py`**: The core simulation integrating TDM and the Bass model. Visualizes agent opinion trajectories and the composition of positive/negative attitude segments.
2. **`sensitivity_analysis.py`**: Executes sensitivity analyses for network hub influence (`m`), filter bubble strength (`b`), and opinion tolerance thresholds. Includes functionality to calculate Effective Sample Size accounting for Lag-1 autocorrelation.
3. **`cross_sectional_ttest.py`**: Extracts cross-sectional data at the final step (`T=150`) to avoid the time-series autocorrelation trap. Performs Welch's t-test (accounting for unequal variances) across independent trials and generates boxplots.

## Requirements
* Python 3.8+
* `numpy`
* `pandas`
* `networkx`
* `matplotlib`
* `scipy`

```bash
pip install numpy pandas networkx matplotlib scipy

# 1. Run the base simulation
python simulation_base.py

# 2. Run the sensitivity analysis
python sensitivity_analysis.py

# 3. Run independent trials and statistical tests (Recommended for paper)
python cross_sectional_ttest.py


Expected Outputs
opinion_dynamics.png: A time-series graph showing individual agent opinion trajectories.

attitude_composition.png: Trends in the number of active positive, negative, and neutral agents.

boxplot_ttest.png: A boxplot illustrating the final distribution of positive agents under different filter bubble conditions, including the p-value calculated via Welch's t-test.

Citation
This code is implemented and extended based on the following academic framework:

Fujii, M. "A Unified Model of Asymmetric Brand Attitude Formation: Integrating Trust-Distrust Dynamics and Innovation Diffusion." (Please update with actual publication details once available)

License
This project is licensed under the MIT License - see the LICENSE file for details.
