# Mathematical_definitions
This is the supplementary file for the paper "Scaling While Privacy Preserving: A Comprehensive Synthetic Tabular Data Generation and Evaluation in Learning Analytics".

The file will provide mathematical definitions for resemblance, utility, and privacy metrics that are used in this paper.

# Resemblance

### (1)	Difference in pairwise correlation. 
Pairwise correlation is a measure of correlation between two expressions and is used to measure the difference between the results of two expressions(Dandekar et al., 2017). It is commonly calculated by computing the covariance between variables and then normalizing by the product of their standard deviations. The mathematical definition for Difference in pairwise correlation is as follows (Hair et al., 2016):

 >Let X_1, X_2, ..., X_n be a set of variables, and Y_1, Y_2, ..., Y_n be another set of variables. The Difference in pairwise correlation (Δρ) is calculated as:

> Δρ = (1/n) * Σ [ρ_{X_iY_i} - ρ_{X_i}ρ_{Y_i}]

> Where:
> - ρ_{X_iY_i} is the correlation coefficient between variables X_i and Y_i.
> - ρ_{X_i} is the average correlation coefficient of variable X_i with the other X variables.
> - ρ_{Y_i} is the average correlation coefficient of variable Y_i with the other Y variables.
> - n is the number of variables.

> This represents the average difference between the pairwise correlation of each variable pair and the product of their average within-group correlations.

In this article, we adopted the approach by (Zhao et al., 2021). We first computed the pairwise correlation matrices of the columns in the real and synthetic datasets. Pearson correlation coefficient and the Theil uncertainty coefficient were used to measure the correlation between continuous features and between categorical features, respectively (Zhao et al., 2021). Finally, the difference between the pairwise correlation matrices of the real and synthetic datasets was calculated (Zhao et al., 2021). The dython library(https://pypi.org/project/dython/) is used to computed pairwise correlation.

### (2)	Jensen-Shannon divergence (JSD)
   The Jensen–Shannon divergence (JSD) is a symmetrized and smoothed version of the Kullback–Leibler divergence. It is defined by (Nielsen, 2019)
 
 > JSD(P, Q) = (1/2) * (D_KL(P || M) + D_KL(Q || M))

 > - D_KL(Q || M)) represents the Kullback-Leibler divergence from distribution P to distribution Q.
 > - P and Q are the two probability distributions being compared.
 > - M is the average distribution, defined as (P+Q)/2

### (3) Wasserstein distance (WD)

![image](https://github.com/ql909/mathematical_definitions/assets/108169831/7b64ead0-18cc-4d5c-9f23-416344aeba9a) (Horan , 2021)



# Utility
Machine learning utility evaluation

### (1)	Random forest classifier

Random forest classifier is an ensemble learning methods used for classification, regression and performs tasks by combining a large number of decision trees to reach a single output (Schonlau, M., & Zou, R. Y, 2020). 

The training algorithm for random forests applies the general technique of bootstrap aggregating, or bagging. Bagging is a technique that involves creating multiple subsets of the original dataset through random sampling with replacement. Given a training set X = x1, ..., xn with responses Y = y1, ..., yn, bagging repeatedly (B times) selects a random sample with replacement of the training set and fits trees to these samples.

For b = 1, ..., B {B (number of trees or samples) is a hyperparameter that can be tuned.}:
      1.	Sample, with replacement, n training examples from X, Y; call these Xb, Yb.
      2.	Train a classification or regression tree fb on Xb, Yb.
After training for all B trees, predictions for unseen samples x' are made by averaging the predictions from all the individual regression trees on x': For regression tasks, the predictions are averaged, 
<img width="147" alt="image" src="https://github.com/ql909/mathematical_definitions/assets/108169831/15342878-e293-42c4-b153-d10589308dec"> while for classification tasks, the majority vote is taken.

### (2)	Multinomial logistic regression
Logistic regression is a supervised machine learning algorithm that predicts the likelihood of an outcome, event, or observation to perform binary classification tasks (Maalouf, M.,2011). It’s referred to as regression because it is the extension of linear regression but is mainly used for classification problems.
Below is an example logistic regression equation (Brownlee, J. ,2023): 

 > y = e^ (b0 + b1*x) / (1 + e^ (b0 + b1*x))

Where y is the predicted output, b0 is the bias or intercept term and b1 is the coefficient for the single input value (x). Each column in your input data has an associated b coefficient (a constant real value) that must be learned from your training data.

### (3) Multi-layer perceptron (MLP)

MLP is a feed forward neural network supplement. It has three layers: input, output, and hidden. The input layer receives the signal to be processed. The output layer is in charge of tasks like prediction and classification. The true computational engine of the MLP is an arbitrary number of hidden layers sandwiched between the input and output layers (Abirami et al., 2020).

The computations taking place at every neuron in the output and hidden layer are as follows,

 > 	ox=Gb2+W2hx ………………………………. (1)

 >  hx=Φx=sb1+W1x …………………………… (2)

with bias vectors b (1), b (2); weight matrices W (1), W (2) and activation functions G and s. The set of parameters to learn is the set θ = {W (1), b (1), W (2), b (2)}. Typical choices for s include tanh function with tanh(a) = (ea − e− a)/ (ea + e− a) or the logistic sigmoid function, with sigmoid(a) = 1/(1 + e− a).


# Privacy
### (1) Distance to Closest Record.

Distance to Closest Record for a given individual s in S as the minimum distance between s and every original individual o in O (Minieri, 2022):
   
    > 𝐷𝐶𝑅(s) = 𝑚𝑖𝑛 𝑑(s,o) for each o∈O
    
   DCR(s) = 0 means that s is an identical copy (clone) of at least one real individual in the original dataset O.
    
### (2)  Nearest Neighbour Distance Ratio (NNDR). 

The nearest neighbor distance ratio, or ratio test, finds the nearest neighbor to the feature descriptor and the second nearest neighbor to the feature descriptor and divides the two (Lieberman, 2023). The formula can be computed as
   
    > NNDR=d1/d2
    
    where d1 is the nearest neighbor distance and d2 is the second nearest neighbor distance.

   According to (Cover & Hart, 1967), the Nearest Neighbor Distance (NND) is a metric that represents the distance from a data point to its nearest neighbor in a dataset. The mathematical definition of Nearest Neighbor Distance for a point x is:

   > NND(x) = min_{y ≠ x} Distance(x, y)

   Here:

   - NND(x) represents the Nearest Neighbor Distance for the data point x
   - min_{y ≠ x} denotes finding the minimum distance over all points y in the dataset except x
   - Distance(x, y)  is the distance metric (e.g., Euclidean distance) between points x and y

    In our paper, we used Euclidean distance in NNDR.
   
### (3)  Membership interface attack (MIA)

Membership inference attack (MIA) when an attacker tries to identify if real records have been used to train the synthetic data generator approach. For this paper, we used the MIA model in (Hernadez et al., 2023)' paper. See details in figure 1 and figure 2 below.

<img width="850" alt="Screenshot 2023-12-13 at 15 45 04" src="https://github.com/ql909/mathematical_definitions/assets/108169831/148e7cfa-2854-4caa-85bf-d281d3374b9e">

                                          figure 1: MIA model (Hernadez et al., 2023)

<p align="center">
  <img width="401" alt="Screenshot 2023-12-13 at 15 46 58" src="https://github.com/ql909/mathematical_definitions/assets/108169831/98ccecb3-8009-4519-8947-35649e4f22e4">

</p>


                                              figure 2: MIA Algorithm  (Hernadez et al., 2023)
## Reference
- Abirami, S., & Chitra, P. (2020). Multilayer Perceptron - an Overview | ScienceDirect Topics. Www.sciencedirect.com. www.sciencedirect.com/topics/computer-science/multilayer-perceptron
- Brownlee, J. (2016). Logistic Regression for Machine Learning. Machine Learning Mastery. https://machinelearningmastery.com/logistic-regression-for-machine-learning/
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21–27. https://doi.org/10.1109/tit.1967.1053964
- Dandekar, A., Zen, R. A. M., & Bressan, S. (2017). Comparative Evaluation of Synthetic Data Generation Methods. Semantic Scholar. https://www.semanticscholar.org/paper/Comparative-Evaluation-of-Synthetic-Data-Generation-Dandekar-Zen/03f0de0a14ae76548bb8614e2aa5e6abfd143a84
- Hair, J., Anderson, R., Black, B., & Babin, B. (2016). Multivariate Data Analysis. Pearson Higher Ed.
- Hernadez, M., Epelde, G., Alberdi, A., Cilla, R., & Rankin, D. (2023). Synthetic Tabular Data Evaluation in the Health Domain Covering Resemblance, Utility, and Privacy Dimensions. Methods of Information in Medicine, 62(Suppl 1), e19–e38. https://doi.org/10.1055/s-0042-1760247
- Horan , C. (2021). Wasserstein Distance and Textual Similarity. Neptune.ai. https://neptune.ai/blog/wasserstein-distance-and-textual-similarity
- Lieberman, A. (2023). Computer Vision Project. Sites.cc.gatech.edu. https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj2/html/alieberman3/index.html
- Maalouf, M. (2011). Logistic regression in data analysis: an overview. International Journal of Data Analysis Techniques and Strategies, 3(3), 281. https://doi.org/10.1504/ijdats.2011.041335
- Minieri, A. (2022). Synthetic Data for Privacy Preservation - Part 2. Clearbox.ai. https://www.clearbox.ai/blog/2022-06-07-synthetic-data-for-privacy-preservation-part-2
- Nielsen, F. (2019). On the Jensen–Shannon Symmetrization of Distances Relying on Abstract Means. Entropy, 21(5), 485. https://doi.org/10.3390/e21050485
- Schonlau, M., & Zou, R. Y. (2020). The random forest algorithm for statistical learning. The Stata Journal: Promoting Communications on Statistics and Stata, 20(1), 3–29. https://doi.org/10.1177/1536867x20909688
- Zhao, Z., Kunar, A., Birke, R., & Chen, L. Y. (2021). CTAB-GAN: Effective Table Data Synthesizing. Proceedings of Machine Learning Research.

