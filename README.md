# Attractor-Chaos
Predicting complex chaotic behaviors using Radial Basis Function Neural Networks 

Built on the work of [rssalessio](https://github.com/rssalessio/PytorchRBFLayer)


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

### **In the case for this project Inverse Multiquadric RBF is used**
