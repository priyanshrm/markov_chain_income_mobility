# Simulation of Economic Mobility: A Markov Chain Approach

**How quickly can households move up the income ladder, and how much can policy change that?**

This project models a country's households as moving between three income classes (Low, Middle, High) from one year to the next. It uses a Markov chain to simulate India's income distribution over 25 years, then adds a single "policy strength" parameter, **k**, to test what happens when reforms make upward mobility easier.

📄 **Paper:** [`markov_chain_income_mobility_policy.pdf`](markov_chain_income_mobility_policy.pdf)
✍️ **Author:** Priyansh Raj Mahendra (December 2025)

---

## The question in plain English

Economists often ask whether a society is *mobile* (a child born poor can realistically become middle class) or *sticky* (where you start is largely where you stay). Mobility is hard to observe directly, but it can be modelled.

This project asks:

1. If we know the yearly odds of moving between income classes, what does the income distribution look like in 10 or 25 years?
2. How sensitive is that outcome to the assumed odds?
3. If a policy (education, credit access, redistribution) raised the odds of moving up, how much would the 25-year picture change?

---

## Headline results

All scenarios start from the same distribution: **99.2% Low, 0.8% Middle, 0.004% High** (116,097 households, ADIS 2019 data). Shares after 25 years:

| Scenario | Low | Middle | High |
|---|---:|---:|---:|
| First-pass matrix (derived from Ranganathan et al., 2016) | 12.7% | 71.8% | 15.4% |
| **Realistic matrix** (sticky; the paper's baseline) | **68.4%** | 26.8% | 4.8% |
| Realistic + weak policy (k = 0.10) | 66.0% | 28.5% | 5.5% |
| Realistic + medium policy (k = 0.25) | 62.5% | 30.9% | 6.6% |
| Realistic + strong policy (k = 0.50) | 57.2% | 34.2% | 8.5% |

---

## What you can learn from this

**1. Small yearly probabilities compound into huge differences.**
Going from the first-pass matrix to the realistic one changes each class's yearly "stay put" probability by only a few percentage points (roughly 85–95% → 97–98%). Over 25 years, that moves the Low-income share from 13% to 68%. In a Markov chain, *stickiness* matters more than almost anything else.

**2. Realistic assumptions matter more than model sophistication.**
The first-pass model predicts a near-frictionless society, which contradicts the empirical evidence on India (low intergenerational mobility, caste barriers, real downward mobility). The paper rebuilds the matrix from that evidence rather than accepting the first output.

**3. Policy helps, but gradually.**
Even the strongest intervention modelled (+50% to every upward transition probability) lowers the 25-year Low-income share by about 11 percentage points. Reforms that shift probabilities modestly do not transform a sticky economy within a generation.

**4. Population growth alone does not change the shares.**
When growth is applied equally to all classes, absolute numbers rise but income shares stay identical. Mobility, not demographics, is what reshapes the distribution.

---

## Background in 60 seconds

**Markov chain.** A model where the next state depends only on the current state. Here, a household's income class next year depends only on its class this year.

**Transition matrix `P`.** A table of yearly probabilities. Entry `P[i][j]` is the chance of moving from class *i* to class *j*. Each row sums to 1.

**Simulation.** If `v₀` is today's distribution across classes, then the distribution after *t* years is:

```
v_t = v₀ · Pᵗ
```

**Sticky economy.** One where the diagonal of `P` is close to 1: most households stay in their class. The extreme case is the identity matrix, where nobody ever moves.

**Steady state.** The distribution a Markov chain settles into after enough years, regardless of where it started.

---

## How the model works

### 1. Income classes

Defined by annual household income (India):

| Class | Annual household income |
|---|---|
| Low ("Aspirers") | ₹1.25 – 5 lakh |
| Middle | ₹5 – 30 lakh |
| High | ₹30 lakh+ |

The paper also converts these to per-capita monthly income (using an average household size of 3.6) and shows how sensitive the thresholds are to household size.

### 2. Transition matrices

**First-pass matrix.** Built from a published study (Ranganathan et al., 2016) that found 31% of the poorest-quintile households stayed there over 7 years (2004-05 to 2011-12), and 47% of the richest stayed. Taking the 7th root converts these to rough yearly persistence rates of 84.6% and 89.7%. Middle-class persistence (95%) is an assumption.

**Realistic matrix.** No study publishes a ready-made 3×3 income transition matrix for India, so this one is constructed from the qualitative evidence: low and stable intergenerational mobility, caste barriers, high inequality, and meaningful downward mobility. It is illustrative rather than estimated.

|  From \ To | Low | Middle | High |
|---|---:|---:|---:|
| **Low** | 0.980 | 0.019 | 0.001 |
| **Middle** | 0.025 | 0.965 | 0.010 |
| **High** | 0.001 | 0.024 | 0.975 |

*How to read it:* of every 1,000 low-income households this year, about 980 stay Low, 19 move to Middle, and 1 jumps to High. Note that a middle-class household is more likely to fall (2.5%) than to rise (1.0%).

### 3. Population growth

India's growth rate (World Bank, 2000–2024) has been falling at about 0.045 percentage points per year. The model starts at 0.89% growth and lets it decay by 4.5% each year, reaching about 0.29% by year 25. Each simulated year applies growth first, then income mobility.

### 4. The k-factor (policy strength)

`k` permanently reshapes the transition matrix:

- **k > 0** multiplies all *upward* transition probabilities by `(1 + k)`.
- **k < 0** multiplies all *downward* transition probabilities by `(1 + |k|)`, representing recessions or regressive policy.
- Each row is then renormalised so it still sums to 1.

Example: at k = 0.50, the Low → Middle probability rises from 1.9% to 2.8% per year, and Middle → High from 1.0% to 1.5%.

---

## Reproduce the headline numbers

```python
import numpy as np

P = np.array([[0.980, 0.019, 0.001],
              [0.025, 0.965, 0.010],
              [0.001, 0.024, 0.975]])
v0 = np.array([115156, 936, 5]) / 116097

def apply_k(P, k):
    Q = P.copy()
    for i in range(3):
        for j in range(3):
            if k > 0 and j > i: Q[i, j] *= 1 + k        # amplify upward moves
            if k < 0 and i > j: Q[i, j] *= 1 + abs(k)   # amplify downward moves
    return Q / Q.sum(axis=1, keepdims=True)             # rows must sum to 1

for k in (0, 0.10, 0.25, 0.50):
    v25 = v0 @ np.linalg.matrix_power(apply_k(P, k), 25)
    print(f"k = {k:.2f}   Low {v25[0]:.1%}   Middle {v25[1]:.1%}   High {v25[2]:.1%}")
```

Output:

```
k = 0.00   Low 68.4%   Middle 26.8%   High 4.8%
k = 0.10   Low 66.0%   Middle 28.5%   High 5.5%
k = 0.25   Low 62.5%   Middle 30.9%   High 6.6%
k = 0.50   Low 57.2%   Middle 34.2%   High 8.5%
```

---

## Limitations

This is a simulation framework, not a forecast. Please read the results with these caveats in mind:

- **The realistic matrix is a judgement call.** It is informed by the literature but not estimated from household panel data. Different defensible matrices give different answers.
- **The first-pass matrix stretches its source.** It collapses five quintiles into three classes and converts a 7-year figure to a yearly one using a root, which is an approximation.
- **The starting distribution drives the results.** The 99.2% Low share follows from the ADIS 2019 data and the income thresholds above. Pew Research estimates roughly 70% / 25% / 5% using different definitions.
- **Memoryless by design.** The model ignores caste, region, education, age and a household's history, all of which affect real mobility.
- **Uniform population growth.** Every class grows at the same rate, which is why shares are unaffected. In reality, fertility differs across income groups.
- **k is stylised.** It scales probabilities by a fixed factor and is not tied to a specific policy, its cost, or its timing.
- **25 years is not the long run.** With the realistic matrix, the distribution is still moving slowly at year 25. Its mathematical steady state is roughly 47% Low / 37% Middle / 17% High, which takes well over a century to approach. The 68.4% figure is a 25-year snapshot, not a final destination.

---

## Repository contents

| File | Description |
|---|---|
| `markov_chain_income_mobility_policy.pdf` | Full paper with derivations, figures and references |

---

## Key references

- Ranganathan, T., Tripathi, A., & Pandey, G. (2016). *Income mobility among social groups in Indian rural households: Findings from the Indian Human Development Survey.*
- Asher, S., Novosad, P., & Rafkin, C. (2024). *Intergenerational mobility in India: Estimates and trends.* American Economic Journal.
- Deshpande, A. (2017). *The Grammar of Caste.* Oxford University Press.
- Pew Research Center (2021). *India's middle class: The size and nature of India's income groups.*
- World Bank (2025). *Population, total – India.* World Bank Open Data.

The full reference list is in the paper.

---

## Citation

```
Mahendra, P. R. (2025). Simulation of Economic Mobility: A Markov Chain Approach.
```
