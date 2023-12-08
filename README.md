# mathematical_definitions
This is the supplementary file for the paper "Scaling While Privacy Preserving: A Comprehensive Synthetic Tabular Data Generation and Evaluation in Learning Analytics".

The file will provide mathematical definitions for resemblance, utility, and privacy metrics that are used in this paper.

# Resemblance

1. Difference in pairwise correlation. Pairwise correlation is a measure of correlation between two expressions and is used to measure the difference between the results of two expressions(Dandekar et al., 2017). It is commonly calculated by computing the covariance between variables and then normalizing by the product of their standard deviations. The mathematical definition for Difference in pairwise correlation is as follows (Hair et al., 2016):

 >Let X_1, X_2, ..., X_n be a set of variables, and Y_1, Y_2, ..., Y_n be another set of variables. The Difference in pairwise correlation (Δρ) is calculated as:

> Δρ = (1/n) * Σ [ρ_{X_iY_i} - ρ_{X_i}ρ_{Y_i}]

> Where:
> - ρ_{X_iY_i} is the correlation coefficient between variables X_i and Y_i.
> - ρ_{X_i} is the average correlation coefficient of variable X_i with the other X variables.
> - ρ_{Y_i} is the average correlation coefficient of variable Y_i with the other Y variables.
> - n is the number of variables.

> This represents the average difference between the pairwise correlation of each variable pair and the product of their average within-group correlations.

In this article, we adopted the approach by (Zhao et al., 2021). We first computed the pairwise correlation matrices of the columns in the real and synthetic datasets. Pearson correlation coefficient and the Theil uncertainty coefficient were used to measure the correlation between continuous features and between categorical features, respectively (Zhao et al., 2021). Finally, the difference between the pairwise correlation matrices of the real and synthetic datasets was calculated (Zhao et al., 2021). The dython library(https://pypi.org/project/dython/) is used to computed pairwise correlation.

2. Jensen-Shannon divergence (JSD)
   The Jensen–Shannon divergence (JSD) is a symmetrized and smoothed version of the Kullback–Leibler divergence. It is defined by (Nielsen, 2019)
 
 > JSD(P, Q) = (1/2) * (D_KL(P || M) + D_KL(Q || M))

 > - D_KL(Q || M)) represents the Kullback-Leibler divergence from distribution P to distribution Q.
 > - P and Q are the two probability distributions being compared.
 > - M is the average distribution, defined as (P+Q)/2

4. Wasserstein distance (WD)

![image](https://github.com/ql909/mathematical_definitions/assets/108169831/7b64ead0-18cc-4d5c-9f23-416344aeba9a) (Horan , 2021)



# Utility
1. machine learning utility evaluation


# Privacy
1. Distance to Closest Record (DCR). Distance to Closest Record for a given individual s in S as the minimum distance between s and every original individual o in O (Minieri, 2022):
   
    > 𝐷𝐶𝑅(s) = 𝑚𝑖𝑛 𝑑(s,o) for each o∈O
    
   DCR(s) = 0 means that s is an identical copy (clone) of at least one real individual in the original dataset O.
    
3. Nearest Neighbour Distance Ratio (NNDR)
4. Membership interface attack (MIA)

## Reference
