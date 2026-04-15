# Optimal_TA_Frequency_Selection

## Overview

This work introduces a **SHAP-based reliability framework** that automatically selects the optimal temporal aggregation (TA) frequency in multivariate time series.  
The method combines **coherence analysis** (consistency of SHAP values across correlated features) and **sensitivity analysis** (stability under input perturbations) into a unified cost function.

The optimal aggregation frequency minimizes this reliability cost, providing a balance between **model interpretability** and **robustness**.

This repository includes a **reproducible implementation** of the proposed framework applied to the *Seagoing Ship* dataset.

## Additional Experimental Results

The appendix results omitted from the paper due to page limitations are provided here.

### Reliability of Predictive Models

For SHIP 3 (Figure \ref{fig:ship3}), ``speed" remained the dominant predictor across all TA frequencies. However, at the 10-minute and 20-minute frequencies, where the highest $R^2$ scores were achieved (0.937 and 0.935, respectively), the models relied on entirely different features. For instance, ``air\_pressure" contributed marginally (0.38) at 10 minutes but dropped to 0.03 at other frequencies, while variables like ``distance" and ``pressure\_compression" appeared and disappeared inconsistently.

<p align="center">
  <img src="figures/ship3.png" width="700"/>
</p>

