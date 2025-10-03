# Scripts for Unified FLUXes (UFLUX) ensemble publication in SciData [DRAFT]

Terrestrial ecosystems regulate climate by absorbing about one-third of anthropogenic CO2 emissions. Monitoring carbon,
water, and energy fluxes is essential for understanding ecosystem responses to climate change. However, existing flux datasets
lack sufficient spatial resolution and consistency needed for fragmented landscapes like UK agricultural areas. This study
presents the Unified FLUXes (UFLUX) ensemble, a globally consistent dataset of gross primary productivity, evapotranspiration,
and sensible heat fluxes derived from eddy covariance data, satellite observations, and machine learning. UFLUX comprises
∼ 60 ensemble members across multiple spatial and temporal scales: global (monthly, 0.25◦), Europe (daily, 0.25◦; biannual,
100 m), and UK (daily, 100 m). Validation against eddy covariance (EC) measurements shows UFLUX captures over 80% of flux
variability, with low mean absolute errors, reproducing climate responses and interannual patterns in line with existing literature,
though uncertainties in net carbon flux remain. UFLUX holds promise for supporting cross-scale climate policymaking and
actions, providing valuable insights for land management and carbon sequestration efforts aimed at a carbon-neutral future.

The code is wrriten in Python (with some comments explaining the code generated with the assitatance of GenAI in a responsible way) 

The default machine learning model is DeepForest 21 but with switches avaible to use like XGBoost and Random Forest

This repository includes two scritps:
- training the machine leanring model at site level data and applying them to grid data to produce products
- process the UFLUX ensmeble products to analyse and presetn, i.e., figures and tables of the manuscropt.

The Data for this repo are on Zenodo
The products are also on Zenodo:

-----

## Overview

The **UFLUX ensemble** (v1) dataset offers multiscale terrestrial carbon flux upscaling products, generated using **machine learning** models. It integrates **satellite-based vegetation proxies** (e.g., vegetation index and solar-induced fluorescence) with **climate reanalysis** like ERA5, and is trained against **eddy covariance** observations (e.g., FLUXNET, ICOS, and Ameriflux). This dataset includes five core flux components:

- Gross Primary Production (GPP)

- Ecosystem Respiration (RECO)

- Net Ecosystem Exchange (NEE)

- Sensible Heat Flux (H)

- Latent Energy Flux (LE)

## Background and Methodology
The **Unified FLUXes (UFLUX)** initiative is a data-driven, machine learning-based platform designed to upscale eddy covariance (EC) flux measurements from tower sites to the global scale. It aims to answer pressing questions about how effectively terrestrial ecosystems are managed under climate change.

Key innovations of UFLUX include:

- `Consistent Upscaling Framework`: Harmonizes flux upscaling across spatial/temporal scales and multiple flux types (GPP, RECO, etc.) using deep decision tree-based methods, better suited than conventional neural networks for EC flux data.

- `Hybrid Explainable ML`: Combines black-box ML with ecological interpretability through residual learning, offering both predictive power and new scientific insight (UFLUXv2).

- `Uncertainty Quantification`: Employs sampling space completeness to assess model uncertainty in a transparent, robust manner.

- `Multisource Integration`: Leverages complementary strengths of vegetation proxies (e.g., NIRv, SIF) and climate data (e.g., ERA5) to represent carbon dynamics more comprehensively than single-source approaches.

- `Superior Gap-Filling`: Originally developed as a global EC flux gap-filling tool, UFLUX improves accuracy by up to 30% and reduces uncertainty by as much as 70% compared to traditional methods.

- `High Performance`: Achieves strong predictive accuracy, with global-scale R² > 0.8 for RECO and ≈0.9 for GPP, while being computationally efficient enough to run on a standard laptop.

- `Community Adoption`: Already used by other global upscaling projects, highlighting its reliability and impact.

## Applications
UFLUX is ideal for studying the interactions between land management, climate change, and carbon fluxes, particularly in improving global estimates of GPP and RECO by addressing biases in EC measurements.

## Resources

UFLUX Website: [https://sites.google.com/view/uflux](https://https://sites.google.com/view/uflux)

Code Repository: [https://github.com/soonyenju/uflux](https://github.com/soonyenju/uflux)

Code Repository for Journal Publication: [https://github.com/songyanzhu/UFLUX-ensemble_scidata](https://github.com/songyanzhu/UFLUX-ensemble_scidata)

Technical & Descriptive Publication: [https://doi.org/10.1080/01431161.2024.2312266](https://doi.org/10.1080/01431161.2024.2312266)



