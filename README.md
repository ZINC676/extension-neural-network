# Extension Neural Network (ENN)

EN / [CN](./README(CN).md)
## 1. Overview

This is an extension neural network (ENN) training and testing tool designed using Python. It is primarily used to solve linear classification problems. Compared to BP neural networks, its training requires significantly less data and time, and the inference speed is extremely fast, with low memory usage.
The code mainly refers to the replication of this paper:

**Wang M H, Hung C P. Extension neural network and its applications[J]. Neural Networks, 2003, 16(5-6): 779-784**
## 2. Usage Steps
### 2.1 Preparing the Dataset

First, collect the data according to the principle of extension neural networks. The specific format can be referred to the files in the dataset folder. Note that each line is a sample data, and each feature data is separated by spaces. The last data on each line is the class of the sample data.
### 2.2 Starting Training & Testing

After organizing the data, put the dataset in the dataset folder, and modify the relevant parameters in ./cfg/dataset/yaml according to your own dataset and training needs. After completing the appropriate parameter adjustments, you can run main.py to perform ENN training and testing. The specific training and testing results will be presented in the form of charts and terminal output.
## 3. Remarks

There is also a testBP3.py file in the code folder. Users can run this program on the same dataset to compare the training speed and accuracy of ENN and BP neural networks (ps: remember to modify the parameters after if __name__ == "__main__": in testBP3.py).
