# Chapter 1: Deep Learning Basics

Welcome to the foundational chapter of the course. Here, we establish the essential theoretical and practical building blocks of Deep Learning using **PyTorch**.

## 🎯 Learning Objectives
In this section, we transition from theory to implementation. Our core goals are:
*   **Fundamentals:** Understanding key definitions and terminology in Deep Learning.
*   **PyTorch Mastery:** Getting comfortable with Tensor operations and the PyTorch ecosystem.
*   **Architecture Construction:** Designing and building your first neural network architectures.
*   **Optimization Insights:** Analyzing computational graphs to visualize how weights are updated and optimized during training.

## 🍕 All the Slices
*  **1-Basics:** in the first part, we explain all the basic concepts of tensor's, pytorch commends, and necessary operations
*  **2-NN_Scratch:** we continue with the forward and backward passes in a Neural Net, and then a real-world problem

## 📂 Contents
This folder contains the materials and practical exercises for the Basic module:

*   `pytorch_basic.py`: A custom helper module containing essential utility functions used across the exercises.
*   `model_weights/`: A directory containing the saved weights of the trained models and some of their structure and architect. These ensure you can reproduce the results without retraining from scratch.
*   `utils/`: All the utility function that are needed in the exercises, they will be automatically downloaded in the exercises

## 🛠 Prerequisites
Before running the notebooks, ensure you have the necessary dependencies installed:
```python
pip install torch torchvision matplotlib sklearn numpy
