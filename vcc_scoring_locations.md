# VCC Scoring Metrics: Code Implementation Locations

This document provides the precise code locations for the three official scoring metrics used in the Virtual Cell Challenge (VCC). The analysis is based on a thorough review of the `cell-eval` repository and a direct comparison with the formulas and descriptions published on the official VCC evaluation webpage.

The findings confirm that the `cell-eval` repository contains the exact implementations for all three metrics.

---

## 1. Perturbation Discrimination Score (PDS)

-   **File**: `src/cell_eval/metrics/_anndata.py`
-   **Function**: `discrimination_score(data: PerturbationAnndataPair, metric: str = "l1", ...)`

### Verification

The implementation in this function exactly matches the PDS description on the VCC website:
- It defaults to using the Manhattan (`l1`) distance.
- It calculates pseudobulk expression profiles for all perturbations.
- For each predicted perturbation, it computes the distance to all true perturbations, finds the rank of the correct match, and normalizes it to a score between 0 and 1 using the formula `1 - rank / total_perturbations`.

---

## 2. Mean Absolute Error (MAE)

-   **File**: `src/cell_eval/metrics/_anndata.py`
-   **Function**: `mae(data: PerturbationAnndataPair, ...)`

### Verification

This function directly implements the MAE metric as described:
- It calls a generic helper function that uses `sklearn.metrics.mean_absolute_error`.
- The calculation is performed on the pseudobulk expression profiles of the predicted and real data, which is consistent with the formula on the VCC website.

---

## 3. Differential Expression Score (DES)

The DES logic is the most complex and is implemented in the `DEComparison` class, which is called by higher-level metric functions.

-   **File**: `src/cell_eval/_types/_de.py`
-   **Class**: `DEComparison`
-   **Method**: `compute_overlap(...)`

### Verification

This method contains the complete and exact algorithm for the DES metric as specified by the VCC:
1.  **Denominator Definition**: The effective number of genes to compare, `k_eff`, is set to the number of significant genes in the **real** data (`real_genes.size`). This matches the denominator in the DES formula.
2.  **Prediction Truncation**: The list of predicted significant genes is explicitly truncated to the size of `k_eff` via `pred_subset = pred_genes[:k_eff]`. This correctly implements the crucial condition for cases where a model predicts more DEGs than exist in the ground truth.
3.  **Final Calculation**: The final score is calculated as `np.intersect1d(real_subset, pred_subset).size / k_eff`. This is the precise formula for the VCC Differential Expression Score: the size of the intersection of the (potentially truncated) predicted set and the true set, divided by the size of the true set.
