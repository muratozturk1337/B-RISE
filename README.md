# B-RISE 
RISE algorithm reimplemented with a new calculation of the importance scores using the Banzhaf value. The code is based on the original RISE implementation by Petsiuk et al.
The main difference is in the way the importance scores are calculated. Instead of using the original RISE method, we use the Banzhaf value to calculate the importance scores for each pixel. This allows us to capture the interactions between pixels and provide a more accurate explanation of the model's predictions.

# The code is structured as follows:
- `brise.py`: Contains the implementation of the B-RISE algorithm
- `evaluation.ipynb`: Contains few examples of insertion and deletion evaluation of the B-RISE and RISE algorithms on the MNIST dataset and RESNET50 model with few images.
- `B_RISE_code.ipynb`: Explains how the B-RISE algorithm works, although it is not very easy to follow just from that notebook.

# TODO 
- Test more images and models to evaluate the performance of the B-RISE algorithm compared to the original RISE method.
