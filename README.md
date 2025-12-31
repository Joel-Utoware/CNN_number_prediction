# CNN-number-prediction
## Project overview
### What is a convolutional neural network?
A convolutional neural network (CNN) is a type of machine learning model that is often used in image recognition.
### What data was used in this project?
* The model was created using a Modified National Institute of Standards (MNIST) dataset containing images with numbers 0 to 9 written on them
* Additionally the Windows application called Paint was used to create 3 additional images to test model working correctly
### How was this project completed?
* The training and testing MNIST datasets were downloaded and preprocessed to be used in the model
* Using the PyTorch library, a 12-layer CNN model was made to classify these images
* data augmentation and regularization was applied to improve model generalization
* The model was then trained and evaluated over 10 epochs and the best model was saved
* It achieved a validation accuracy of 99.5% 
* The best model was then used to classify the three images made in Paint where it was 100\% accurate

## MLOps Implementation (WIP)
* Testing (Pytest), Containerization (Docker), Deployment (Cloud Run)
## Live Service (WIP)
* A clickable link to the live hosted API and a clear explanation of how to use it.
