[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2404.00618)

# A Multi-Branched Radial Basis Network Approach to Predicting Complex Chaotic Behaviours
Predicting complex chaotic behaviors using Radial Basis Function Neural Networks 




Radial Basis Functions (RBFs) are a class of mathematical functions widely used in various fields, including machine learning and computational mathematics. These functions are defined based on the distance or similarity between a point and a center, often in a multidimensional space.

They are defined as:

1. **Gaussian RBF**:
φ(r) = exp(-r^2 / (2 * σ^2))

where *r* is the distance between the input point and the center, and *σ* is a parameter controlling the width of the Gaussian.

2. **Multiquadric RBF**:
φ(r) = sqrt(1 + (r / σ)^2)

where *r* is the distance between the input point and the center, and *σ* is a parameter controlling the shape of the function.

4. **Inverse Multiquadric RBF**:
φ(r) = 1 / sqrt(1 + (r / σ)^2)

where *r* is the distance between the input point and the center, and *σ* is a parameter controlling the shape of the function.

4. **Thin Plate Spline RBF**:
φ(r) = r^2 * log(r)

where *r* is the distance between the input point and the center.

## Proposed Model Architecture 

![rbfmulti](https://github.com/chungimungi/A-Multi-Branched-Radial-Basis-Network-Approach-to-Predicting-Complex-Chaotic-Behaviours/assets/90822297/14590960-a8ca-4712-8b22-f54e958dfa8e)


### Trained using Inverse Multiquadratic
#### Training Parameters
- epochs = 2000
- batch_size = 512
- Prediction Steps = 100

## Results

[GIFs](https://drive.google.com/drive/folders/1l-uGRRqru-eUcQGWe73-svW-konhBoBa?usp=sharing)

**Consider citing this work if you find it useful**



