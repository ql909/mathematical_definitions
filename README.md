# mathematical_definitions
This is the supplementary file for the paper "Towards privacy at scale: A Comprehensive Synthetic Tabular Data Generation and Evaluation in Learning Analytics".

The file will provide mathematical definitions for resemblance, utility, and privacy metrics that are used in this paper.

# Resemblance

1. Difference in pairwise correlation. Pairwise correlation is a measure of correlation between two expressions and is used to measure the difference between the results of two expressions(Dandekar et al., 2017). It is commonly calculated by computing the covariance between variables and then normalizing by the product of their standard deviations. The mathematical definition for Difference in pairwise correlation is as follows (Hair et al., 2016):

Let X_1, X_2, ..., X_n be a set of variables, and Y_1, Y_2, ..., Y_n be another set of variables. The Difference in pairwise correlation (Δρ) is calculated as:

Δρ = (1/n) * Σ [ρ_{X_iY_i} - ρ_{X_i}ρ_{Y_i}]

Where:
- ρ_{X_iY_i} is the correlation coefficient between variables X_i and Y_i.
- ρ_{X_i} is the average correlation coefficient of variable X_i with the other X variables.
- ρ_{Y_i} is the average correlation coefficient of variable Y_i with the other Y variables.
- n is the number of variables.

This represents the average difference between the pairwise correlation of each variable pair and the product of their average within-group correlations.

In this article, we adopted the approach by (Zhao et al., 2021). We first computed the pairwise correlation matrices of the columns in the real and synthetic datasets. Pearson correlation coefficient and the Theil uncertainty coefficient were used to measure the correlation between continuous features and between categorical features, respectively (Zhao et al., 2021). Finally, the difference between the pairwise correlation matrices of the real and synthetic datasets was calculated (Zhao et al., 2021). The dython library(https://pypi.org/project/dython/) is used to computed pairwise correlation.

2. Jensen-Shannon divergence (JSD)
3. Wasserstein distance (WD)


# Utility
1. machine learning utility evaluation


# Privacy
1. Distance to Closest Record (DCR)
2. Nearest Neighbour Distance Ratio (NNDR)
3. Membership interface attack (MIA)

## Reference
