# CONVOLUTION NEURAL NETWORK IMAGE CLASSIFICATION 📷

This project aims to explore / demonstrate the ability of Neural Networks to successfully categorise and classify image data. The data is provided by Microsoft with the intention of helping to improve CAPTCHA tests.

📈 The model achieves a validation accuracy of ~0.81 (81%)
### Background from Microsoft: ###

_Web services are often protected with a challenge that's supposed to be easy for people to solve, but difficult for computers. Such a challenge is often called a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof). HIPs are used for many purposes, such as to reduce email and blog spam and prevent brute-force attacks on web site passwords. Asirra (Animal Species Image Recognition for Restricting Access) is a HIP that works by asking users to identify photographs of cats and dogs. This task is difficult for computers, but studies have shown that people can accomplish it quickly and accurately. Asirra is unique because of its partnership with Petfinder.com, the world's largest site devoted to finding homes for homeless pets. They've provided Microsoft Research with over three million images of cats and dogs, manually classified by people at thousands of animal shelters across the United States._

Data Source: _https://www.microsoft.com/en-us/download/details.aspx?id=54765_



### Repo Structure 📂
- PetImages
  - Cats: ~12500 images of cats
  - Dogs: ~12500 images of dogs
- main.py: Python script to process and fit image data to Neural Network model.

### Model Architecture
Layer 1: Convolutional Layer
- Neurons: 64
- Activation Function: Rectified Linear Unit (ReLU) is utilized to introduce non-linearity, enhancing the model's capability to learn complex patterns.
- Dimensionality Reduction: Applies Max Pooling to compress the spatial dimensions of the output, effectively reducing the computational complexity while preserving essential features.


Layer 2: Convolutional Layer
- Neurons: 64
- Activation Function: ReLU, continuing to facilitate the learning of intricate structures within the data.
- Dimensionality Reduction: Max Pooling is employed once again to further condense the output dimensions, streamlining feature representation.


Layer 3: Fully Connected Layer (Dense Layer)
- Activation Function: ReLU, ensuring the network's capacity to process and relay non-linear relationships to the output layer.

Output Layer: Final Classification Layer
- Neurons: 1, tailored for binary classification tasks.
- Activation Function: Sigmoid, chosen for its ability to map the final neuron's output to a probability score between 0 and 1, indicative of the two classes (e.g., 'dog' or 'cat').


Loss Function: Binary Crossentropy
- Purpose: This loss function is selected due to the binary nature of the classification task. It quantifies the difference between the predicted probabilities and the actual binary labels (dog or cat), guiding the model towards accurate predictions through optimization.


<img align="left" width=250 src="PetImages/Cat/0.jpg"> <img align="right" width=250 src="PetImages/Dog/0.jpg">
