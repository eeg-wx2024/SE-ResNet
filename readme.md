基于您提供的`readme.md`文件，我为您重新生成了一个更清晰、更结构化的README文件。它将原始文件中的关键修改点进行了解释和归纳，使其更易于理解。

-----

### Statistics-fused learning for classification of randomized EEG trials

**Project Overview**

This repository contains the implementation details and configuration for the deep learning framework described in our paper, "Statistics-fused learning for classification of randomized EEG trials". This project focuses on improving the classification of randomized EEG trials by introducing several key modifications to a ResNet backbone.

**Key Features and Modifications**

The following is a breakdown of the key elements and their implementation in the code, as referenced in the provided logs and notes.

1.  **Merged Training (MT)**

      * **Purpose**: To improve the quality of the training data and enhance learning effectiveness.
      * **Implementation**: During the training process, two samples are merged into a single sample for training. The validation process uses a single sample.
      * **Code Reference**: `train_N = 2`, `val_N = 1`.

2.  **Approximated Discrete Cosine Transform (ADCT)**

      * **Purpose**: To introduce features from the frequency domain, providing a richer representation of the EEG data and enhancing discriminability.
      * **Implementation**: A portion of the EEG patches are replaced with their ADCT features. The optimal replacement ratio was found to be 25%.
      * **Code Reference**:
          * `main.py`: `x = x_group(x)`
          * `data_utils.py`: `x = group_replace_patch(x, ratio=0.25)`

3.  **ELU Activation Function**

      * **Purpose**: To replace the standard ReLU activation function with Exponential Linear Units (ELU) to improve robustness against noise and accelerate model convergence.
      * **Implementation**: The activation function within the ResNet model is set to ELU.
      * **Code Reference**: `resnet(..., act_name="elu", ...)`

4.  **Attention Mechanism**

      * **Purpose**: To enable more focused learning by guiding the network to concentrate on the most discriminative feature channels. Squeeze-and-Excitation Networks (SE-Net) were chosen for their effectiveness and low complexity.
      * **Implementation**: The attention module within the ResNet model is set to SE-Net.
      * **Code Reference**: `resnet(..., attn_name="se", ...)`

5.  **Statistics-fused Learning**

      * **Purpose**: To coordinate the outputs of four parallel ResNet-based classifiers and produce a final, robust classification decision.
      * **Implementation**: The system groups the input patches and uses a voting mechanism based on the classification results of the four individual classifiers.
      * **Code Reference**:
          * `main.py`: `x = to_patch(x, replace=True)`
          * `main.py`: `x = x_group(x)`
