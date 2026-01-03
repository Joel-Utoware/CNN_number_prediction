# CNN-number-prediction

## Live Service 
Link to try the model:
[try the model](https://mnist-api-1097413744504.us-central1.run.app)

## Setup to optionaly run via Docker
    git clone https://github.com/Joel-Utoware/CNN_number_prediction.git
    docker build -t mnist-api .
    docker run -p 8080:8080 mnist-api


## Project overview
![Architecture](images/Architecture.png)
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
### Training results
![model_training_results](images/metric_results.png)
* epoch_loss is the average training loss
* val_loss is the average validation loss 
### Testing
The project includes a suite of automated tests using `pytest` to ensure model reliability and input validation.

<details>
<summary>View Test Results</summary>

```bash
================================================== test session starts ==================================================
platform win32 -- Python 3.11.5, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Users\joelb\Documents\GitHub\CNN_number_prediction
configfile: pytest.ini
plugins: anyio-4.12.0
collected 5 items                                                                                                        

tests\test_model.py .....                                                                                          [100%]

=================================================== 5 passed in 6.16s =================================================== 
```

</details>

### FastAPI Integration
Have now implementd Fast API and HTML code to allow users to be able to upload images for the model to try and predict. If live service is unavailble users can use the local site when running FastAPI or the /docs endpoint.

<details>
<summary>succesful_upload</summary>
  
![succesful_upload](images/success.png)

</details>

<details>
  
<summary>no_upload</summary>

![no_upload](images/no_upload.png)

</details>

<details>
  
<summary>wrong_file</summary>

![wrong_file](images/wrong_file_type.png)

</details>

For a higher chance of a succesful prediction, images must a whole number between 0 and 9  on a white background. 

<details>
  
<summary>For example</summary>

![number 1](tests/data/num_1.png)

![number 5](tests/data/num_5.png)

</details>

### Containerization (Docker)
The project was containerised using docker and docker compose was used to deploy the mlfow and FastApi at the same time.
<details>
  
<summary>MlFlow and FastAPI together</summary>

![MlFlow and FastAPI together](images/docker_duo.png)

</details>

<details>
  
<summary>Recieving request</summary>

![recieving request](images/dockerports.png)

</details>

<details>
  
<summary>Docker console</summary>

![docker console](images/docker_image.png)

</details>

### Deployment (Cloud Run)
The model is available to try while it is still hosted on Google cloud services. 

<details>
  
<summary>Google cloud console</summary>

![Google cloud console](images/Gcloud_use.png)

</details>

<details>
  
<summary>Working on Google cloud</summary>

![working on Google cloud](images/Gcloud_image.png)

</details>


