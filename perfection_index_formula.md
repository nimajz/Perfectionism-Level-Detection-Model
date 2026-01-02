# Perfection Index: Mathematical Formulation

## Overview

The Perfection Index (PI) is a continuous score between 0 and 1 that quantifies a player's perfectionistic behavior in an educational game. It is computed as a weighted linear combination of five behavioral components, where each component is derived from session-level aggregated features.

## Main Formula

The Perfection Index for a session $s$ is computed as:

$$\text{PI}(s) = \text{clip}\left( \sum_{i=1}^{5} w_i \cdot C_i(s), 0, 1 \right)$$

where:
- $w_i$ are component weights such that $\sum_{i=1}^{5} w_i = 1$
- $C_i(s)$ are normalized component scores for session $s$
- $\text{clip}(x, 0, 1)$ ensures the final score is bounded in $[0, 1]$

## Component Weights

The five components and their weights are:

| Component | Weight $w_i$ | Symbol |
|-----------|--------------|--------|
| Accuracy | 0.4 | $w_1$ |
| Hover Duration | 0.2 | $w_2$ |
| Retry Behavior | 0.2 | $w_3$ |
| Settings Consistency | 0.1 | $w_4$ |
| Exploration Thoroughness | 0.1 | $w_5$ |

## Component 1: Accuracy Score ($C_1$)

**Weight:** $w_1 = 0.4$

**Formula:**
$$C_1(s) = A(s)$$

where $A(s)$ is the raw accuracy for session $s$:

$$A(s) = \frac{\sum_{q=1}^{Q} \mathbf{1}[q \text{ answered}] \cdot \mathbf{1}[\text{answer}_q = \text{correct}]}{\sum_{q=1}^{Q} \mathbf{1}[q \text{ answered}]}$$

**Explanation:**
- $A(s)$ is the proportion of correctly answered questions out of all attempted questions in session $s$
- $Q$ is the total number of available questions
- $\mathbf{1}[\cdot]$ is the indicator function
- This component is **not normalized** and is used directly (already in $[0,1]$ range)

**Rationale:** Perfectionists strive for correctness, so higher accuracy directly increases the Perfection Index.

---

## Component 2: Hover Duration Score ($C_2$)

**Weight:** $w_2 = 0.2$

**Formula:**
$$C_2(s) = \text{norm}(\bar{H}(s))$$

where:
- $\bar{H}(s)$ is the mean hover duration for session $s$
- $\text{norm}(\cdot)$ is min-max normalization across all sessions

**Raw Feature Computation:**
$$\bar{H}(s) = \frac{1}{|\mathcal{E}_s|} \sum_{e \in \mathcal{E}_s} h_e$$

where:
- $\mathcal{E}_s$ is the set of events in session $s$ with non-null hover duration
- $h_e$ is the hover duration (in milliseconds) for event $e$
- If no hover events exist, $\bar{H}(s) = 0$

**Normalization:**
$$\text{norm}(x) = \begin{cases}
0.5 & \text{if } \max_S - \min_S = 0 \\
\frac{x - \min_S}{\max_S - \min_S} & \text{otherwise}
\end{cases}$$

where:
- $\min_S = \min_{s' \in S} \bar{H}(s')$ (minimum across all sessions $S$)
- $\max_S = \max_{s' \in S} \bar{H}(s')$ (maximum across all sessions $S$)

**Special case:** If all sessions have the same hover duration (constant feature), all scores are set to 0.5 to avoid division by zero.

**Explanation:** Perfectionists spend more time examining interface elements before interacting, indicating careful consideration. The normalization ensures this component is on the same scale as others.

---

## Component 3: Retry Behavior Score ($C_3$)

**Weight:** $w_3 = 0.2$

**Formula:**
$$C_3(s) = \text{norm}\left(\frac{\text{clip}(R_{\text{total}}(s), 0, 10)}{10}\right)$$

**Raw Feature Computation:**

First, compute the total revisits:
$$R_{\text{total}}(s) = R_{\text{room}}(s) + R_{\text{fqid}}(s)$$

where:
- $R_{\text{room}}(s) = |\{r : \text{count}_s(r) > 1\}|$ (number of rooms visited more than once)
- $R_{\text{fqid}}(s) = |\{f : \text{count}_s(f) > 1\}|$ (number of objects/characters visited more than once)
- $\text{count}_s(x)$ counts occurrences of element $x$ in session $s$

**Clamping and Scaling:**
$$R_{\text{scaled}}(s) = \frac{\text{clip}(R_{\text{total}}(s), 0, 10)}{10} = \begin{cases}
0 & \text{if } R_{\text{total}}(s) \leq 0 \\
1 & \text{if } R_{\text{total}}(s) \geq 10 \\
\frac{R_{\text{total}}(s)}{10} & \text{otherwise}
\end{cases}$$

**Normalization:**
$$C_3(s) = \text{norm}(R_{\text{scaled}}(s))$$

where normalization is applied across all sessions as defined in Component 2.

**Explanation:** Moderate retries indicate persistence and attention to detail (perfectionism), but excessive retries may indicate confusion rather than perfectionism. The clamping to $[0, 10]$ followed by division by 10 creates a soft cap, then min-max normalization scales it to $[0, 1]$.

---

## Component 4: Settings Consistency Score ($C_4$)

**Weight:** $w_4 = 0.1$

**Formula:**
$$C_4(s) = 0.5 \cdot \text{norm}(R_{\text{hq}}(s)) + 0.3 \cdot \text{norm}(R_{\text{fullscreen}}(s)) + 0.2 \cdot \text{norm}(R_{\text{music}}(s))$$

**Raw Feature Computation:**

For each setting type $t \in \{\text{hq}, \text{fullscreen}, \text{music}\}$:

$$R_t(s) = \frac{\text{count}_s(\text{mode}_t(s))}{\sum_{v \in V_t} \text{count}_s(v)}$$

where:
- $\text{mode}_t(s)$ is the most frequent value of setting $t$ in session $s$
- $\text{count}_s(v)$ counts occurrences of value $v$ in session $s$
- $V_t$ is the set of all possible values for setting $t$
- If no settings data exists, $R_t(s) = 0$

**Sub-component Weights:**
- High-quality (HQ) setting: weight 0.5
- Fullscreen setting: weight 0.3  
- Music setting: weight 0.2

**Normalization:** Each $R_t(s)$ is normalized independently across all sessions using the min-max normalization defined in Component 2.

**Explanation:** Perfectionists tend to maintain consistent quality settings throughout their session, preferring high-quality options. The ratio measures how consistently the player uses their preferred setting value. Each setting is normalized separately before weighted combination.

---

## Component 5: Exploration Thoroughness Score ($C_5$)

**Weight:** $w_5 = 0.1$

**Formula:**
$$C_5(s) = \text{norm}(U_{\text{rooms}}(s))$$

**Raw Feature Computation:**
$$U_{\text{rooms}}(s) = |\{r : r \in \text{rooms}_s\}|$$

where:
- $\text{rooms}_s$ is the set of unique rooms (room_fqid values) visited in session $s$
- If no room data exists, $U_{\text{rooms}}(s) = 0$

**Normalization:**
$$C_5(s) = \text{norm}(U_{\text{rooms}}(s))$$

where normalization is applied across all sessions as defined in Component 2.

**Explanation:** Perfectionists tend to explore thoroughly, visiting more unique locations in the game world. This component rewards comprehensive exploration behavior.

---

## Final Clipping

The final Perfection Index is clipped to ensure it remains in $[0, 1]$:

$$\text{PI}(s) = \text{clip}\left( \sum_{i=1}^{5} w_i \cdot C_i(s), 0, 1 \right) = \begin{cases}
0 & \text{if } \sum_{i=1}^{5} w_i \cdot C_i(s) < 0 \\
1 & \text{if } \sum_{i=1}^{5} w_i \cdot C_i(s) > 1 \\
\sum_{i=1}^{5} w_i \cdot C_i(s) & \text{otherwise}
\end{cases}$$

In practice, since all weights sum to 1 and all components are in $[0, 1]$, the sum is already bounded, but clipping provides numerical stability.

---

## Complete Expanded Formula

Substituting all components into the main formula:

$$\text{PI}(s) = \text{clip}\Bigg( 0.4 \cdot A(s) + 0.2 \cdot \text{norm}(\bar{H}(s)) + 0.2 \cdot \text{norm}\left(\frac{\text{clip}(R_{\text{total}}(s), 0, 10)}{10}\right) $$

$$+ 0.1 \cdot \left(0.5 \cdot \text{norm}(R_{\text{hq}}(s)) + 0.3 \cdot \text{norm}(R_{\text{fullscreen}}(s)) + 0.2 \cdot \text{norm}(R_{\text{music}}(s))\right)$$

$$+ 0.1 \cdot \text{norm}(U_{\text{rooms}}(s)), 0, 1 \Bigg)$$

---

## Implementation Notes

1. **Missing Data Handling:** All raw features default to 0 if no data is available for that feature in a session.

2. **Normalization Domain:** All min-max normalizations are computed across the **entire dataset** (all sessions) before computing individual session scores. This ensures fair comparison across sessions.

3. **Constant Feature Handling:** If a feature has zero variance (all sessions have the same value), normalization returns 0.5 for all sessions to avoid division by zero.

4. **Data Types:** 
   - Accuracy is computed from binary labels (0 = incorrect, 1 = correct)
   - All counts are non-negative integers
   - All ratios are in $[0, 1]$
   - Time-based features (hover duration) are in milliseconds

5. **Computational Complexity:** The normalization operations require two passes: first to compute min/max across all sessions, then to normalize each session. This is acceptable since the number of sessions is typically much smaller than the number of events.

