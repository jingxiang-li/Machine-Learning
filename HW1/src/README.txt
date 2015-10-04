Author: Jingxiang Li (5095269)
Email: lixx3899@umn.edu

=============================================================================================================================

This project contains python scripts for CSCI 5525 HW1

There are 4 executable python scripts:
    1. Fisher: Fisher Discriminant Analysis + Multivariate Gaussian Classifier for multi-class classification problem
    2. SqClass: Least Square Discriminant for multi-class classification problem
    3. logisticRegression: Logistic Regression for two-class classification problem
    4. naiveBayesDiscrete: Naive Bayes Model with Univariate Gaussian Approximation for multi-class classification problem

Requirements for running python scripts:
    python 3.4
    numpy 1.9.3
    scipy 0.16.0
    matplotlib 1.4.3 (optional, only for 'plot_naivebayes_logreg.py')

Requirements for the dataset:
    1. Must be in .csv format
    2. Header is not allowed in the dataset
    3. Row for cases, column for features
    4. The first column should be the indicator of classes
    5. Missing value is not allowed in the dataset
    6. Logistic Regression in this project only supports two-class problems

To run the scripts, use the following commands in the terminal
    ./Fisher /path/to/dataset.csv crossval
        Example: ./Fisher ../res/MNIST-1378.csv 10
    ./SqClass /path/to/dataset.csv crossval
        Example: ./SqClass ../res/MNIST-1378.csv 10
    ./naiveBayesGaussian /path/to/dataset.csv num_splits train_percent
        Example: ./naiveBayesGaussian ../res/spam.csv 100 "5 10 15 20 25 30"
    ./logisticRegression /path/to/dataset.csv num_splits train_percent
        Example: ./logisticRegression ../res/spam.csv 100 "5 10 15 20 25 30"


