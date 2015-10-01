Author: Jingxiang Li (5095269)
Email: lixx3899@umn.edu

=============================================================================================================================

This project contains python scripts for CSCI 5525 HW1

There are 4 executable python scripts:
    Fisher: Fisher Discriminant Analysis + Multivariate Gaussian Classifier for multi-class classification problem
    SqClass: Least Square Discriminant for multi-class classification problem
    logisticRegression: Logistic Regression for two-class classification problem
    naiveBayesDiscrete: Naive Bayes Model with Univariate Gaussian Approximation for multi-class classification problem

Requirements for running python scripts:
    python 3.4
    numpy 1.9.3
    scipy 0.16.0
    matplotlib 1.4.3 (optional, only for 'plot_naivebayes_logreg.py')

To run the scripts, use the following commands in the terminal
    ./Fisher /path/to/dataset crossval
    ./SqClass /path/to/dataset crossval
    ./naiveBayesGaussian /path/to/dataset num_splits
    ./logisticRegression /path/to/dataset num_splits

Requirements for dataset:
    1. must be .csv file
    2. no header dataset
    3. the first column should be the indicator of classes
    4. Missing value is not allowed in the dataset
    5. Logistic Regression in this project only supports two-class problems
