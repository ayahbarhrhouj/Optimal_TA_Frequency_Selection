# Optimal_TA_Frequency_Selection

## Overview

This work introduces a **SHAP-based reliability framework** that automatically selects the optimal temporal aggregation (TA) frequency in multivariate time series.  
The method combines **coherence analysis** (consistency of SHAP values across correlated features) and **sensitivity analysis** (stability under input perturbations) into a unified cost function.

The optimal aggregation frequency minimizes this reliability cost, providing a balance between **model interpretability** and **robustness**.

This repository includes a **reproducible implementation** of the proposed framework applied to the *Seagoing Ship* dataset.

