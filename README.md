# Domain-Adversarial Training of Neural Networks (DANN) Implementation

## Overview

This project implements Domain-Adversarial Training of Neural Networks (DANN) by modifying the ResNet50 Convolutional Neural Network (CNN). DANN is a domain adaptation technique that enables a neural network to learn features that are invariant across different domains.

## Features

- **Domain-Adversarial Training:** The project utilizes the DANN technique to train a neural network, specifically modifying the ResNet50 architecture for domain adaptation.

- **ResNet50 Modification:** The ResNet50 CNN architecture is customized to incorporate domain-adversarial training components.

## Getting Started

### Prerequisites

- [Python](https://www.python.org/) and [pip](https://pip.pypa.io/)
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)

### Installation

1. Clone the repository:

   git clone [Domain-Adaptation](https://github.com/ugendar07/Domain-Adaptation.git)

## Usage
- Prepare your dataset for domain adaptation, ensuring that it includes samples from different domains.

- Configure the model parameters and dataset paths in the configuration files.

- Train the DANN model
  
- Evaluate the trained model


## Datasets
- MNIST-USPS:

  - MNIST Dataset: [MNIST](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist)
  - USPS Dataset: [USPS](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#mnist)
OfficeHome:

  - OfficeHome Dataset: [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html)
 
## Acknowledgments
The implementation is based on the Domain-Adversarial Neural Network (DANN) algorithm proposed by [ Ganin et al.](https://arxiv.org/abs/1505.07818).
Special thanks to the creators of [ResNet](https://arxiv.org/abs/1512.03385) for the base CNN architecture.


## Contact
For questions or support, please contact ugendar07@gmail.com.
