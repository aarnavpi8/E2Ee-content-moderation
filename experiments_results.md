# Robustness Tests Walkthrough

We have implemented and executed a suite of rigorous tests to evaluate the scientific robustness of the Linear and MLP models. Below are the structured tables, statistical test results, and comparative graphs.

---

## 1. 5-Fold Cross-Validation (SMS Dataset)

We evaluated both models across three feature dimensions $d \in \{64, 256, 1024\}$ using Stratified 5-Fold Cross-Validation. Below is the performance (Mean $\pm$ SD) of the float and integer-quantized classifiers.

| Feature Dimension ($d$) | Model | Accuracy | True Positive Rate (TPR / Spam Recall) | False Positive Rate (FPR / Ham Blocked) | Precision | F1-Score |
|---|---|---|---|---|---|---|
| **64** | Linear (Float) | 0.9094 ± 0.0082 | 0.5235 ± 0.0495 | 0.0309 ± 0.0063 | 0.7254 ± 0.0439 | 0.6068 ± 0.0403 |
| **64** | Linear (Quantized) | 0.9094 ± 0.0082 | 0.5235 ± 0.0495 | 0.0309 ± 0.0063 | 0.7254 ± 0.0439 | 0.6068 ± 0.0403 |
| **64** | MLP (Float) | 0.9279 ± 0.0047 | 0.7282 ± 0.0524 | 0.0412 ± 0.0061 | 0.7332 ± 0.0199 | 0.7294 ± 0.0257 |
| **64** | MLP (Quantized) | 0.9277 ± 0.0045 | 0.7282 ± 0.0524 | 0.0414 ± 0.0059 | 0.7321 ± 0.0184 | 0.7288 ± 0.0254 |
| **256** | Linear (Float) | 0.9661 ± 0.0032 | 0.8554 ± 0.0235 | 0.0168 ± 0.0051 | 0.8890 ± 0.0303 | 0.8712 ± 0.0111 |
| **256** | Linear (Quantized) | 0.9661 ± 0.0032 | 0.8554 ± 0.0235 | 0.0168 ± 0.0051 | 0.8890 ± 0.0303 | 0.8712 ± 0.0111 |
| **256** | MLP (Float) | 0.9670 ± 0.0042 | 0.8768 ± 0.0284 | 0.0191 ± 0.0041 | 0.8776 ± 0.0216 | 0.8768 ± 0.0159 |
| **256** | MLP (Quantized) | 0.9670 ± 0.0042 | 0.8768 ± 0.0284 | 0.0191 ± 0.0041 | 0.8776 ± 0.0216 | 0.8768 ± 0.0159 |
| **1024** | Linear (Float) | 0.9790 ± 0.0043 | 0.8808 ± 0.0320 | 0.0058 ± 0.0011 | 0.9592 ± 0.0070 | 0.9180 ± 0.0181 |
| **1024** | Linear (Quantized) | 0.9790 ± 0.0043 | 0.8808 ± 0.0320 | 0.0058 ± 0.0011 | 0.9592 ± 0.0070 | 0.9180 ± 0.0181 |
| **1024** | MLP (Float) | 0.9794 ± 0.0038 | 0.8849 ± 0.0248 | 0.0060 ± 0.0022 | 0.9582 ± 0.0150 | 0.9199 ± 0.0152 |
| **1024** | MLP (Quantized) | 0.9795 ± 0.0040 | 0.8862 ± 0.0267 | 0.0060 ± 0.0022 | 0.9583 ± 0.0150 | 0.9206 ± 0.0160 |

### 📊 Cross-Validation Metrics Comparison
![Cross-Validation Comparison](/home/agnibh/.gemini/antigravity/brain/937cae6b-188c-4d28-ae44-9990291470a8/cv_comparison.png)

> [!NOTE]
> - Integer quantization results in negligible degradation compared to the float baselines.
> - The MLP shows a significant performance gain over the Linear model at low dimension ($d=64$), especially in Spam Recall (TPR) and F1-score. As the dimension rises to $d=1024$, the gap between Linear and MLP narrows.

---

## 2. McNemar's Significance Test ($d=256$)

To test whether the difference in error rates between the quantized Linear model and the quantized MLP model is statistically significant, we aggregated out-of-fold predictions over the entire SMS dataset:

| | MLP Correct | MLP Incorrect |
|---|---|---|
| **Linear Correct** | 5343 | 42 (b) |
| **Linear Incorrect** | 47 (c) | 142 |

- **McNemar Chi-Squared Statistic (with continuity correction)**: $0.1798$
- **P-Value**: $0.6716$

> [!IMPORTANT]
> The p-value ($p \approx 0.6716$) is significantly greater than $0.05$. Therefore, we cannot reject the null hypothesis; the predictive performance difference between the quantized Linear and MLP models at $d=256$ is **not statistically significant** on the SMS corpus.

---

## 3. Out-of-Distribution Generalization (Enron Corpus)

We evaluated the models trained on the SMS dataset against an out-of-distribution (OOD) Enron email corpus (5,000 stratified samples).

| Dimension ($d$) | Model | Accuracy | True Positive Rate (TPR / Recall) | False Positive Rate (FPR) | Precision | F1-Score |
|---|---|---|---|---|---|---|
| **64** | Linear (Float) | 0.5112 | 0.8156 | 0.7932 | 0.5070 | 0.6253 |
| **64** | Linear (Quantized) | 0.5112 | 0.8156 | 0.7932 | 0.5070 | 0.6253 |
| **64** | MLP (Float) | 0.5386 | 0.6616 | 0.5844 | 0.5310 | 0.5891 |
| **64** | MLP (Quantized) | 0.5388 | 0.6616 | 0.5840 | 0.5311 | 0.5892 |
| **256** | Linear (Float) | 0.5314 | 0.7912 | 0.7284 | 0.5207 | 0.6280 |
| **256** | Linear (Quantized) | 0.5314 | 0.7912 | 0.7284 | 0.5207 | 0.6280 |
| **256** | MLP (Float) | 0.5322 | 0.7504 | 0.6860 | 0.5224 | 0.6160 |
| **256** | MLP (Quantized) | 0.5322 | 0.7508 | 0.6864 | 0.5224 | 0.6161 |
| **1024** | Linear (Float) | 0.5252 | 0.7740 | 0.7236 | 0.5168 | 0.6198 |
| **1024** | Linear (Quantized) | 0.5252 | 0.7740 | 0.7236 | 0.5168 | 0.6198 |
| **1024** | MLP (Float) | 0.5394 | 0.3796 | 0.3008 | 0.5579 | 0.4518 |
| **1024** | MLP (Quantized) | 0.5394 | 0.3796 | 0.3008 | 0.5579 | 0.4518 |

### 📊 OOD Generalization Comparison
![OOD Generalization Comparison](/home/agnibh/.gemini/antigravity/brain/937cae6b-188c-4d28-ae44-9990291470a8/enron_generalization.png)

> [!WARNING]
> - There is a severe degradation in accuracy (dropping to ~51-54%) across all models due to the distribution shift from short SMS text to formal Enron emails.
> - While TPR remains relatively high, FPR is also extremely high (often >70%), showing that the model classifies the majority of messages as spam under the new distribution.

---

## 4. Adversarial Evasion with 95% Bootstrap Confidence Intervals ($d=256$)

We computed the evasion rates at budget $k=3$ with 95% Bootstrap Confidence Intervals (1000 resamples) on the correctly blocked spam subset.

| Model | Attack Type | Evasion Rate | 95% CI Lower Bound | 95% CI Upper Bound |
|---|---|---|---|---|
| **Linear (Quantized)** | Synonym | 6.72% | 4.93% | 8.81% |
| **Linear (Quantized)** | Homoglyph | 0.00% | 0.00% | 0.00% |
| **Linear (Quantized)** | Whitespace | 0.00% | 0.00% | 0.00% |
| **MLP (Quantized)** | Synonym | 7.80% | 5.91% | 9.81% |
| **MLP (Quantized)** | Homoglyph | 0.00% | 0.00% | 0.00% |
| **MLP (Quantized)** | Whitespace | 0.00% | 0.00% | 0.00% |

### 📊 Adversarial Evasion with CIs
![Adversarial Evasion CIs](/home/agnibh/.gemini/antigravity/brain/937cae6b-188c-4d28-ae44-9990291470a8/evasion_bootstrap_ci.png)

> [!TIP]
> - The normalisation defenses completely neutralize homoglyph and whitespace-based evasion attacks (0% evasion rate).
> - Synonym attacks retain a low but positive evasion rate, with the MLP showing a slightly higher rate (7.80%) than the Linear model (6.72%), though the confidence intervals overlap significantly.
