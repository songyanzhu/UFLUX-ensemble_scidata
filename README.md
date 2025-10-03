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



