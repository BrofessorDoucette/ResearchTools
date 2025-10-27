This repository is filled mostly with python scripts that I used for data analysis while I was a graduate student working with Professor Allison Jaynes. 

The case studies themselves were usually handled in jupyter notebooks while the heavy duty lifting was done in the source .py files and then imported into the notebooks. However, as always, slowly over time the repository has sort of degraded in quality and is probably in proper need of a refactoring. There are files in here for working with a machine learning model on an imbalanced dataset, simulating radial diffusion using the 1-D Fokker Planck equation, investigating alfven waves, and plotting radial phase space density profiles.

The project I had spent the most time on was predicting the chorus power measured by EMFISIS on the Van Allen Probes, using low-earth orbit electron flux measurements from POES and geomagnetic indices (Kp, SME). This was a hard problem because of the heteroscedasticity of the raw data, and also the lack of conjunctions between EMFISIS and POES. I was unable to fix the problem to my satisfaction using transformations (box-cox, log, etc), weights, or a loss function that lets the variance be variable (see https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html). This meant the residuals were not normal and the dataset was heavily imbalanced.

This is the average distribution that EMFISIS measured during the Van Allen Probes Lifetime. The model I developed should essentially be able to accurately predict deviations from this average using the extra information from POES and indices.

<img src="Average_Chorus_Vs_L_MLT.png" alt="drawing" width="400"/>


