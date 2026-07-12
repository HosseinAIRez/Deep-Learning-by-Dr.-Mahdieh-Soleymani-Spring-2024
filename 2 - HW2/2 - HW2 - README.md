# Chapter 2: <br>Deep Learning Classification, CNNs, Segmentation, Object Detection

In this chapter, we move on to deep learning for computer vision. We cover image classification, CNN architectures, semantic segmentation, and object detection.
>**Note:**<br>
>This section of the repository is divided into two parts:
>
>- Theoretical
>- Practical
## ✒️ Theoretical part
In this section, we review and strengthen our theoretical understanding of:
- Batch Normalization: how it works, computational cost benefits and others...
- Dilated Convolution: its complete workflow and formulation
- ROI Alignment: concept and meaning in action
- Convolution Gradient: how the gradient flows through the network
- Vanishing gradient: the main problem in the **really deep** Neural Networks
- MobileNet: it's usecase and upgraded variants of it

## 👨🏾‍💻 Practical Part<br>
Before starting the practical exercises, it is worth mentioning that all trained models and generated DataFrames are stored in the `Results/Models` and `Results/Dataframes` directories so that you can compare your own results with mine.

*  **1-Classification:** in the first part, we will test the reall world datasets
 `cifar 10` and `cifar 100` for classification purposes, and how the architecture
  of a model really affects the Models accuracy.
   <p align="center">
      <img src="cifar 10.png" alt="cifar 10 dataset" width="600">
      <br>
      <em>Figure 1: cifar 10 dataset</em>
    </p>
    <p align="center">
      <img src="cifar 100.png" alt="cifar 100 dataset" width="900">
      <br>
      <em>Figure 2: cifar 100 dataset</em>
    </p>

## 📂 Contents
The `Results` directory contains the following files:

* `Models`:
  * `cifar10_Q1_checkpoint.pth`: final model after the trining on the cifar 10 dataset with 
  approximately 90% accuracy on the test set
  * `cifar100_Q1_checkpoint.pth`: The same pretrained feature extractor with only the final classification 
  layer retrained on CIFAR-100 dataset with approximately 40% accuracy on the test set.
* `Dataframes`:
  * `cifar10_Q1_Classification_test_df.xlsx`& `cifar10_Q1_Classification_train_df.xlsx`
   &  `cifar10_Q1_Classification_valid_df.xlsx` the results of training and evaluation model on 
   the `cifar 10` dataset saved for visualizing purposes
  * `cifar100_Q1_Classification_test_df.xlsx`& `cifar100_Q1_Classification_train_df.xlsx`
   &  `cifar100_Q1_Classification_valid_df.xlsx` the results of training and evaluation model on 
   the `cifar 100` dataset saved for visualizing purposes
   


## 🛠 Prerequisites

Install the required packages before running the notebooks:

```bash
pip install torch torchvision matplotlib scikit-learn numpy pandas
```
