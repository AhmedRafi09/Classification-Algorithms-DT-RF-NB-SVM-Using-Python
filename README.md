### Introduction:
In this project, we are going to do a comparative analysis of some major
classification algorithms, (i.e: Decision Tree, Naïve Bayes, Random Forest, Support
Vector Machine) on a dataset. We use a dataset from Kaggle. The algorithms are
implemented in python using the scikit-learn library and also in weka3.8 tool. We
also evaluated the performances of the algorithms using accuracy, precision, recall,
f1-score etc. performance measure metrics. A comprehensive and supportive
discussion is included to justify the results of the classification algorithms. The
dataset holds some medical insurance data. We wanted to classify if an individual
is a ‘Smoker’ or not based on the other information of the dataset. We have
implemented the four above mentioned classification algorithms to generate a
comparative study of their performance.

### Classification Algorithms:

### Decision Tree Induction:
Decision tree builds classification or regression models in the form of a tree
structure. It breaks down a dataset into smaller and smaller subsets while at the
same time an associated decision tree is incrementally developed. The final result
is a tree with decision nodes and leaf nodes. A decision node has two or more
branches. Leaf node represents a classification or decision. The topmost decision
node in a tree which corresponds to the best predictor called root node. Decision
trees can handle both categorical and numerical data.

The core algorithm for building decision trees, called ID3, employs a top-down,
greedy search through the space of possible branches with no backtracking. ID3
uses Entropy and Information Gain to construct a decision tree.
A decision tree is built top-down from a root node and involves partitioning the
data into subsets that contain instances with similar values (homogenous). ID3
algorithm uses entropy to calculate the homogeneity of a sample. If the sample is
completely homogeneous the entropy is zero and if the sample is an equally
divided it has entropy of one. ID3 uses information gain as its attribute selectionmeasure. Let node N represent or hold the tuples of partition D. The expected
information needed to classify a tuple in D is given by
Info(D) = − ∑ 𝑝𝑖𝑙𝑜𝑔2(𝑝𝑖)

#### Working Mechanism:
- Find the Entropy of the dataset.
- Find the entropy of each of the individual dataset.
- Find the gain.

#### Advantages and disadvantages:
- Simple to understand, interpret and visualize.
- Can handle both numerical and categorical data.
- Nonlinear parameters don’t affect its performance.
- Main disadvantage is over fitting.
- The model can get unstable due to small variation in data.



### Naïve Bayes:
Naïve Bayes Classifier is based on Bayes theorem which gives the conditional
probability of an event A given B. A Naive Bayes classifier assumes that the
presence of a particular feature in a class is unrelated to the presence of any other
feature.
#### Working Mechanism:
- Compute the prior probability.
- Compute conditional probability.
- Compute likelihood.
- Take the max value

#### Advantages and dis advantages:
- Fast and highly scalable algorithm.
- It is a simple algorithm that depends on bunch of counts and also easy to implement.
- It needs less training data
- Not sensitive to irrelevant features.


### Random Forest:

Random Forest algorithm is a tree based supervised classification
algorithm. It is one of the most popular and powerful supervised machine
learning algorithm. Random forest is a method that operates by
constructing multiple decision trees during training phase. In general the
more trees in the forest the more robust the prediction and thus the higher
accuracy. The decision of the majority of the trees is chosen by the random
forest as the final decision. Random Forest algorithm is capable to perform
the regression and classification both task.

#### Working Mechanism:
- Take a random sample of size N with replacement from the data.
- Take a random sample without replacement of the predictors.
- Construct the first CART partition of the data.
- Repeat Step 2 for each subsequent split until the tree is as large as desired. Do not prune.
- Repeat Steps 1–4 a large number of times

#### Advantages and disadvantages:
- Handle the missing values and maintains accuracy for missing data.
- Won’t over fit the model.
- Handle large dataset with higher dimensionality.
- Capable to perform both regression and classification


### Support Vector Machine:

A Support Vector Machine (SVM) is a discriminative classifier formally defined by a
separating hyperplane. In other words, given labeled training data (supervised
learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In this algorithm, we plot each data item as a point in n-dimensional
space (where n is number of features you have) with the value of each feature
being the value of a particular coordinate. Then, we perform classification by
finding the hyper-plane that differentiate the two classes very well.

#### Working Mechanism:
- Create more than hyper-planes to separate the classes.
- Maximize the distances between nearest data point (either class) and hyper-plane to identify the right hyper-plane. This distance is called as Margin.

#### Advantages and dis advantages:
- It works really well with clear margin of separation
- It is effective in high dimensional spaces.
- It is effective in cases where number of dimensions is greater than the number of samples.
- It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.



### Dataset
The dataset is taken from kaggle.com, a machine learning dataset repository. The
link is: https://www.kaggle.com/mirichoi0218/insurance.
The dataset contains 7 columns and 1338 rows. 

The columns are:
- Age
- Sex
- BMI
- Children
- Smoker
- Region
- Charges


These attributes tell us about an individual of his age, if he is a male or female, the
Body Mass Index, if he is a smoker or not, residence region and the medical cost
billed by health insurance.
From the dataset, we want to make a training set and test set to determine if an
individual is a **‘smoker’ or not.**


### Performance Evaluation:

We implemented 4 classification algorithms on the dataset to find out if an
individual is a smoker or not.
According to accuracy, Decision Tree outperforms other algorithms yielding an
accuracy score of 96.7%. The nearest competitor of DT is the Random Forest,
which is also a tree based algorithm, having the accuracy of 96.1%. Then the SVM
was able to predict 94.9% accurately about the person being a smoker whereas
Naïve Bayes performs with an accuracy of 94.3%. All of this accuracy scores are
pretty good in terms of classification.

Sometimes the Accuracy Score does not tell the whole story of the performance of
an algorithm. In our case, we want to determine a smoker. Here comes the other
performance measures: precision, recall and f1-score.
The precision for a class is the number of true positives (i.e. the number of items
correctly labeled as belonging to the positive class) divided by the total number of
elements labeled as belonging to the positive class.
Recall is known as sensitivity of a dataset. Sensitivity (true positive rate) measures
the proportion of actual positives that are correctly identified as such. Recall in this
context is defined as the number of true positives divided by the total number of
elements that actually belong to the positive class.
In a classification task, a precision score of 1.0 for a class C means that every item
labeled as belonging to class C does indeed belong to class C (but says nothing
about the number of items from class C that were not labeled correctly) whereas a
recall of 1.0 means that every item from class C was labeled as belonging to class C
(but says nothing about how many other items were incorrectly also labeled as
belonging to class C).
The two measures are sometimes used together in the F1 Score (or f-measure) to
provide a single measurement for a system.
Here, the recall of SVM is 1. So it performs better to find out a smoker. But overall
performance is better for DT. RF performs nearly as good as DT.

Now, if the algorithm predicts a ‘smoker’ as ‘non-smoker’ it’s a not big deal. But if
it predicts a ‘non-smoker’ as a ‘smoker’ it’s bad. Because, we don’t want to label
someone as ‘smoker’ who does not smoke. Here, we consider the TPR and FPR
rate to consider for the performance. TPR is also known as recall which is discussed
before. The FPR rate is important here.

In our case, it gives us the rate of predicted smoker but actually the person is not a
smoker. So we want such an algorithm that minimizes the FPR rate.According to FPR, DT (0.029) performs best among the four algorithms. RF (0.031)
is nearly as good as DT. And they have quite similar TPR rate. Both the algorithms
are good for solving our problem of finding true smoker, yet, DT performs the best


### Discussion

From the above section, we have learned that Decision Tree and Random Forest
performs better than the other algorithms. These two are Tree Based algorithms.
Naïve Bayes works on Bayes theorem which gives the conditional probability to
calculate prior and posterior probability and then predict an outcome. It yields a
good performance but just not as good as DT, RF or SVM.
Support Vector Machine is very useful to classify tight datasets where there is
necessity of a very sharp margin of separation. And it is effective in high
dimensional spaces. Our dataset is simple and the features are not too edgy to
predict, so we get modest performance.
Generally Random Forest was supposed to perform better than Decision Tree, as
Random Forest is the ensemble model of Decision Tree. But aggregated/ensemble
models are not universally better than their "single" counterparts, they are better
if and only if the single models suffer of instability. With 1338 rows and only 7
columns, the dataset is in a comfortable training sample size situation in which
even a decision tree may get reasonably stable. Ensemble models built on all
variates can be better than their single submodels - but only if the submodels
suffer from instability.
For computational load, a single decision tree will train much more quickly, and
computing a prediction is also much quicker.
So that’s why DT is performing better than RF and any of the four algorithms.


### Conclusion

The comparative analysis of the four major classification algorithms shows good
performance in predicting an individual if he is a smoker or not. All of the four
algorithms, DT, NB, RF, SVM show accuracy over 90% and Decision Tree
outperforms the other algorithms, even the ensemble model Random Forest.
