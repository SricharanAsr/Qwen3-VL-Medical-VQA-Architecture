# Post-Hoc Temperature Scaling & Selective Abstention Mathematical Specification

## 1. Temperature Scaling Formulation (Stage 10)
Given an uncalibrated logit vector $\mathbf{z}_i \in \mathbb{R}^K$ produced by `Qwen3-VL-4B`, standard softmax probabilities $\sigma(\mathbf{z}_i)_k = \frac{\exp(z_{i,k})}{\sum_j \exp(z_{i,j})}$ frequently exhibit pathological overconfidence.

Temperature scaling introduces a scalar parameter $T > 0$ such that:
$$\hat{p}_{i,k}(T) = \frac{\exp(z_{i,k} / T)}{\sum_{j=1}^K \exp(z_{i,j} / T)}$$

### Properties:
1. **Argmax Invariance:** For any $T > 0$, $\arg\max_k \hat{p}_{i,k}(T) = \arg\max_k \sigma(\mathbf{z}_i)_k$. The ranking of clinical predictions is strictly preserved.
2. **Entropy Control:** $T > 1$ increases predictive entropy, softening extreme logits towards uniform distributions.
3. **Parameter Optimization:** $T^*$ is optimized via Negative Log-Likelihood (NLL) on the held-out validation cohort $\mathcal{D}_{\text{val}}$:
   $$T^* = \arg\min_{T > 0} -\sum_{i \in \mathcal{D}_{\text{val}}} \log \hat{p}_{i, y_i}(T)$$

Empirically derived optimal temperature: $T^* = 1.150$.

---

## 2. Expected Calibration Error (ECE)
The calibration error partitions predictions into $M=10$ equally spaced confidence bins $B_m \subset (0, 1]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

- **Pre-Calibration ECE:** $12.14\%$
- **Post-Calibration ECE:** $2.85\%$ ($\Delta = -9.29\%$)

---

## 3. Selective Abstention Decision Boundary (Stage 12)
Let $c(Y \mid X)$ be the length-normalized sequence confidence proxy:
$$c(Y \mid X) = \exp\left( \frac{1}{N} \sum_{t=1}^N \log P_{T^*}(y_t \mid y_{<t}, X) \right)$$

The decision function $g(X)$ with threshold $\tau = 0.60$ is:
$$g(X) = \begin{cases} \text{DISCHARGE} & \text{if } c(Y \mid X) \ge \tau \quad (92.40\% \text{ Coverage}) \\ \text{ABSTAIN} & \text{if } c(Y \mid X) < \tau \quad (7.60\% \text{ Routed to Specialist}) \end{cases}$$