# Extension Neural Network (ENN)

EN / [CN](./README(CN).md)

## 1. Introduction

This is a training and testing tool for Extension Neural Network (ENN), designed with Python language. It is mainly used to predict linear problems. Compared with BP neural network, the data and time required for training are greatly reduced, and the inference speed is extremely fast with less memory consumption.

## 2. Usage

### 2.1 Prepare the Dataset

Firstly, collect the data according to the principle of Evolvable Neural Network. The specific format can refer to files inside `dataset` and `test_dataset` folder. Note that each row represents a sample data, each feature data is separated by a space, and the last data of each row represents the category of the sample data.

### 2.2 Start Training & Testing

After that, put the dataset into the `dataset` folder and run `main.py`. The meaning of the parameters can be found in the adjacent comments.
