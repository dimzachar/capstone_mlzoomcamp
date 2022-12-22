## Capstone Project mlzoomcamp Image Classification 

<p align="center">
<img src="https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/collage.jpg">
</p>


## Table of Contents
 * [Description of the problem](#description-of-the-problem)
 * [Project Objectives](#project-objectives)
 * [Local deployment](#local-deployment)
 * [Production deployment](#production-deployment-with-bentoml)
   * [Docker container](#docker-container)
   * [Cloud deployment](#cloud-deployment)
 * [More](#what-else-can-i-do)


Repo contains the following:

* `README.md` with
  * Description of the problem
  * Instructions on how to run the project
* `notebook.ipynb` a Jupyter Notebook with the data analysis and models
* Script `train.py` (suggested name)
  * Training the final model
* Script `lambda-function.py` for predictions. The script is formatted for deployment on Amazon Web Services' Lambda.
* final model .h5
* Files with dependencies
  * `env_project.yml` conda environment (optional)
* Instructions for Production deployment
  * Video or image of how you interact with the deployed service
* Documentation with code description
* The original dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/vencerlanz09/shells-or-pebbles-an-image-classification-dataset)



## Description of the problem

Written with help of #ChatGPT

Have you ever been to the beach and found yourself wanting to collect either shells or pebbles, but not sure which was which? Or maybe you're in the oil and gas industry and need a quick and accurate way to classify different geological materials? Well, I have the solution for you!

Introducing the Shells or Pebbles dataset – a collection of images specifically designed for binary classification tasks. With this dataset, you'll be able to easily determine whether a certain image is a shell or a pebble.

But the usefulness of this dataset doesn't stop there. In the oil and gas industry, accurately identifying and classifying different materials, including rocks and shells, is crucial for exploration and production activities. By understanding the composition and structure of the earth's layers, geologists can make informed decisions about where to drill for oil and gas.

And for those concerned about the environment, this dataset can also be used to study the impacts of climate change on coastal ecosystems. By analyzing the distribution and abundance of shells and pebbles on beaches, scientists can gain valuable insights into the health of marine life and the effects of human activities.

So whether you're an artist looking to create a beach-themed project or a scientist studying the earth's geological makeup, the Shells or Pebbles dataset has something to offer. With its reliable and accurate classification capabilities, this dataset can help you make better informed decisions and better understand the world around you.


## Project Objectives

Potential objectives for this project include:

* Develop a model that performs well on a binary classification problem.
* Tune the model's hyperparameters to get the best possible accuracy.
	* Used learning rate, droprate as main hyperparameters. Also added data augmentation but due to lack of time and computer resources didn't spend much time on tuning it further. Size of inner layers, img size and other parameters could also be changed by the user.
* Use the callbacks to save the best model weights and and end training if the validation accuracy does not increase after a certain number of epochs.
* Utilize TensorBoard to visualize the training process and find trends or patterns in the data (I didn't make use of this in the end).
* Use the trained model to accurately categorize new photos as Shells or Pebbles.
* Deploy the trained model in a production environment.
* Create comprehensive [Documentation](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Documentation.md) for the project, including a detailed description of the model architecture, training procedure and deployment.
* Display the project's outcomes in a more professional way.


I selected possible best parameters and architecture to achieve a good accuracy. It is possible the architecture is not much suitable or there are other parameters that better fit this problem. It would need more investigation on the dataset and on creation of the model.

## Local deployment

All development was done on Windows with conda.

You can either recreate my environment by
```bash
conda env create -f env_project.yml
conda activate project
```

or do it on your own environment.

Download repo
```bash
https://github.com/dimzachar/capstone_mlzoomcamp.git
```

Notes: 
* You can git clone the repo in Saturn Cloud instead of running it in your own pc. 
* Just make sure you have set it up, see [here](https://github.com/dimzachar/mlzoomcamp/blob/master/Notes/saturn.md). Create secrets for Kaggle in order to download the data.
* You don't need pipenv if you use Saturn Cloud
* See instructions below for more
* You can access the environment here
[![Run in Saturn Cloud](https://saturncloud.io/images/embed/run-in-saturn-cloud.svg)](https://app.community.saturnenterprise.io/dash/o/community/resources?templateId=80e7a844ff5649d2a17552f9aa66628d)


For the virtual environment, I utilized pipenv. 

If you want to use the same venv as me, install pipenv and dependencies, navigate to the folder with the given files:

```bash
cd capstone_mlzoomcamp
pip install pipenv
pipenv shell
pipenv install numpy pandas seaborn jupyter plotly scipy tensorflow==2.9.1 scikit-learn==1.1.3 tensorflow-gpu
```

Before you begin you need to download the data. You can either download them manually from [Kaggle](https://www.kaggle.com/datasets/vencerlanz09/shells-or-pebbles-an-image-classification-dataset) or use the kaggle cli with your API keys (you need to download the kaggle.json from your profile amd paste it in PATH/.kaggle) and extract the files
```bash
kaggle config set -n api.username -v YOUR_USERNAME
kaggle config set -n api.key -v YOUR_API_KEY

kaggle datasets download -d vencerlanz09/shells-or-pebbles-an-image-classification-dataset -p Images
```
![kaggle](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/kaggle.png)

If you run it on Saturn Cloud make sure you are inside <code>/tensorflow/capstone_mlzoomcamp</code>.


This will download the zip file inside folder named Images. Then, unzip it inside this folder manually or using git bash and delete the zip file. Since you are inside capstone_mlzoomcamp folder do
```bash
unzip -q Images/shells-or-pebbles-an-image-classification-dataset.zip -d Images
rm Images/shells-or-pebbles-an-image-classification-dataset.zip
```


Folder structure should now look like this
```
Images
├───Pebbles
└───Shells
```
Now run <code>create_directories</code> script, which will split the images into train, val and test folders (60%,20%,20%) with labels

```bash
pipenv run python create_directories.py
```

The final structure before you train the model should look like this
```
Images
├───test
│   ├───Pebbles
│   └───Shells
├───train
│   ├───Pebbles
│   └───Shells
└───val
    ├───Pebbles
    └───Shells
```


To open the `notebook.ipynb` and see what is inside (optional), run jupyter

```bash
pipenv run jupyter notebook
```




For the evaluation you would need to run <code>train.py</code>. This, will run the train function and construct a ML model with best parameters which will be saved in <code>checkpoints</code> folder (it will be created automatically). The model with highest validation accuracy will be loaded, evaluated (it will return some metrics) and then converted to a Tensorflow Lite model in order to deploy it in the cloud later.
Note: If you run it on a CPU it will take some time (minimum 20 minutes). It is a good idea to use a GPU to speed up the training process. 


```bash
pipenv run python train.py
```

Note: 
* Ignore if you get any warnings (you shouldn't get but in any case) and wait till you see the message <code>Finished</code>. In the end you will have a <code>model.tflite</code> file in the directory. You can also find the best model in .h5 format inside the <code>checkpoints</code> folder.
* If you don't want to run <code>train.py</code> (even though you should) there are files in folder <code>Extra_models</code> in <code>.h5</code> and <code>.tflite</code> format. I have no responsibility if they work (I guess they do).

## Production deployment


### Docker container

To deploy the model locally, follow these steps:

* Install Docker on your system. Instructions can be found [here](https://docs.docker.com/get-docker/).
* Build the Docker image for the model and run the container using the following commands:


```bash
docker build -t model .
docker run -it --rm -p 8080:8080 model:latest
```
then run

```bash
pipenv run python test.py
```

to test it locally using an url. 

The function returns a dictionary with a single key-value pair, where the key is the class label and the value is the prediction value. The class label is "Shells" if the prediction value is greater than or equal to 0.5, or "Pebbles" if the prediction value is less than 0.5. The prediction value is always greater than or equal to 0.5.

For example, if the value of pred is 0.7, the class label will be "Shells" and the prediction value will be 0.7. If the value of pred is 0.3, the class label will be "Pebbles" and the prediction value will be 0.7.

![local](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/testing_local.png)


### Cloud deployment 

In order to deploy it to AWS we push the docker image. Make sure you have an account and install AWS CLI. Instructions can be found [here](https://mlbookcamp.com/article/aws)

First, create a repository on Amazon Elastic Container Registry (ECR) with an appropriate name
![registry2](https://github.com/dimzachar/mlzoomcamp_projects/blob/master/00-midterm_project/Images/Elastic-Container-Registry%20(2).png)

![registry](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/Elastic-Container-Registry.png)

You will find the push commands there to tag and push the latest docker image
![ECR](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/Elastic-Container-push.png)

which you find on your system with

```bash
pipenv run docker images
```

Next, we publish to AWS Lambda.

Go to AWS Lambda, create function, select container image and add a name. Then, browse your image and finally hit create function 
![function](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/Create-function-Lambda.png)

Go to configuration, change timeout to 30 seconds and increase memory RAM (e.g. 1024)
![settings](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/Edit-basic-settings-Lambda.png)

Test the function by changing the event json
![eventjson](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/event-json.png)

Expose the lambda function using API Gateway. Go to API Gateway, select REST API and build a new API
![apigate](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/API-Gateway.png)

Create a new API, give a name
![apigatecreate](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/API-Gateway-Create-API.png)

Create new resource, name it predict
![apiresource](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/api_resource.png)

Create new method, select POST and hit click. Choose Lambda function as integration type and on Lambda function give the name of the function you created and hit save 
![post](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/post.png)

Hit Test, add a JSON document on request body
```bash
 {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Pebbleswithquarzite.jpg/1280px-Pebbleswithquarzite.jpg" }
```

or other image

![posttest](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/post_test.png)
![testjson](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/test-json.png)


Hit Deploy on Actions, select New Stage and give a name

![deployapi](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/deploy-api.png)

Copy the invoke URL, put it in your /test.py file and run it
![testinvoke](https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/test_invoke_url.png)

Make sure you remove/delete everything after testing if necessary. 

Video of cloud deployment 

https://github.com/dimzachar/capstone_mlzoomcamp/blob/master/Extra/shells.mp4

That's a wrap!

### What else can I do?
* Send a pull request.
* If you liked this project, give a ⭐.

**Connect with me:**

<p align="center">
  <a href="https://www.linkedin.com/in/zacharenakis/" target="blank"><img align="center" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" height="30" width="30" /></a>
  <a href="https://github.com/dimzachar" target="blank"><img align="center" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" height="30" width="30" /></a>

  
</p>
           
