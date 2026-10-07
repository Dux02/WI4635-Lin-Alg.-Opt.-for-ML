## Results 2
(a) Implement a function for logistic regression (with regularization) as in the lecture.
(b) Create a random data matrix X and construct an output vector y by generating a random
weight vector w and setting yi = sign(xTi w), where xTi is the i-th row of X. Use a test/train
split and check the performance of ridge and logistic regression for binary classification.
Do you see a large difference in performance between these methods?
(c) Now create a data set (X,y) for binary classification (with X ∈ Rn×d and y ∈ {−1,1}n)
such that, given a test/train split, OLS and ridge perform very badly but logistic regression
performs well. What kind of properties of your data set are responsible for this?