Author: Jingxiang Li (5095269)
Email: lixx3899@umn.edu

=========================================================================

This project contains python scripts for CSCI 5525 HW2

Requirements for running python scripts:
    python 3.4+
    numpy 1.9.1+
    matplotlib 1.4.3+ (optional)

Requirements for the dataset:
    1. Must be in .csv format
    2. Header is not allowed in the dataset
    3. Row for cases, column for features
    4. The first column should be the indicator of classes
    5. Missing value is not allowed in the dataset

========================================================================

To run the scripts, use the following commands in the terminal:

    python3 ./mysmosvm.py /path/to/dataset.csv numruns
        Example: python3 ./mysmosvm.py ../res/MNIST-13.csv 2

    python3 ./mysgdsvm.py /path/to/dataset.csv k numruns
        Example: python3 ./mysgdsvm.py ../res/MNIST-13.csv 20 5

========================================================================

Scripts have been tested in the cselab
