# Qwen3-VL-4B Medical VQA: 15-Stage System Architecture Specification

## Executive Overview
This specification details the end-to-end architecture for the **Medical Visual Question Answering (VQA)** pipeline for Thoracic & Pulmonary Radiography based on the `Qwen3-VL-4B` foundational vision-language model.

---

## 15-Stage Architectural Reference

| Stage | Identifier | Operational Domain | Functional Specification | Mathematical / Parameter Reference |
|---|---|---|---|---|
| **1** | Single-Organ Filtering Engine | Ingestion | Curates PMC-VQA strictly for Chest & Lung radiographs, verifying AP/PA views. | $\text{Organ} = \text{'Pulmonary'}$, PA/AP verified |
| **2** | RadLex Term Canonicalizer | Semantic Preprocessing | Normalizes free-text synonymy to RSNA RadLex Ontology IDs (`RID`). | RadLex RID Mapping (`RID:28491`) |
| **3** | Reasoning Stratification | Taxonomy | 3-tier hierarchy: L1 Recognition, L2 Localization, L3 Complex Diagnosis. | Hierarchical 3-Tier Taxonomy |
| **4** | Open-Ended Prompt Transformer | Prompt Engineering | Strips MCQ cues (A/B/C/D) to prevent option-bias and encourage free-text reasoning. | Option Stripper Regex Engine |
| **5** | NF4 Double Quantization | Model Compression | NormalFloat4 quantization with bfloat16 compute precision on frozen base model. | Base Frozen, VRAM < 6.0 GB |
| **6** | Joint Vision-Language LoRA | Parameter-Efficient PEFT | Low-rank adaptation ($r=32, \alpha=64$) on LLM attention/MLP and vision merger projector. | $r=32, \alpha=64$, 33.5M parameters |
| **7** | Difficulty-Weighted Loss | Multimodal Optimization | Stratified cross-entropy loss multipliers: L1=1.0x, L2=1.5x, L3=2.0x. | $\mathcal{L} = \sum w_i \mathcal{L}_{\text{CE}}$ |
| **8** | 5,000-Step Multimodal Training | Optimization Loop | Cosine annealing schedule with 200 warmup steps using AdamW optimizer. | AdamW, $\eta=2\times 10^{-4}$, 200 warmup |
| **9** | Token Log-Likelihood Proxy | Uncertainty Estimation | Length-normalized geometric mean token confidence metric. | $c(Y \mid X) = \exp\left(\frac{1}{N}\sum_{t=1}^N \log P(y_t)\right)$ |
| **10** | Temperature Scaling Calibration | Calibration | Post-hoc temperature optimization ($T^*=1.150$) resolving overconfidence. | $T^*=1.150, \text{ECE}: 12.14\% \to 2.85\%$ |
| **11** | Verification Re-Prompt Controller | Safety Harness | Bounded Attempt 2 re-prompting with anatomic visual cues for low-confidence L2/L3 queries. | Max 2 attempts, Thoracic cues |
| **12** | Selective Abstention Mechanism | Clinical Triaging | Dual-threshold routing: autonomous discharge vs specialist radiologist referral. | $\tau = 0.60$, 92.40% Coverage, 7.60% Abstain |
| **13** | BioBERT Semantic Similarity | Clinical Evaluation | Cosine embedding similarity benchmark using domain-specific BioBERT. | $\text{Cosine} \ge 0.82$ vs Gold Standard |
| **14** | Automated Jupyter Engine | Reproducibility | Automated headless execution via `MMC.ipynb` pipeline runner. | Headless Papermill execution |
| **15** | PDF Report Generator | Deliverables | Generates publication-ready vector schematics and master technical reports. | `archi.pdf` & Master Report |

---

## Multimodal Loss Formulation

$$\mathcal{L}_{\text{stratified}} = \sum_{i=1}^{M} w(c_i) \cdot \left[ -\sum_{t=1}^{N_i} \log P_\theta(y_{i,t} \mid y_{i,<t}, \mathbf{X}_i) \right]$$

Where:
$$w(c_i) = \begin{cases} 1.0 & \text{for } c_i \in \text{L1: Recognition} \\ 1.5 & \text{for } c_i \in \text{L2: Localization} \\ 2.0 & \text{for } c_i \in \text{L3: Complex Diagnosis} \end{cases}$$

---

## Uncertainty Quantification & Expected Calibration Error

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Post-Hoc Calibrated Softmax:
$$\hat{P}_{i,k} = \frac{\exp(z_{i,k} / T^*)}{\sum_{j} \exp(z_{i,j} / T^*)}, \quad T^* = 1.150$$
