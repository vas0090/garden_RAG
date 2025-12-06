# Gardening RAG 
This repository consists of the documentation, datasets, and future code for creating a RAG system for personal home gardening applications.

Authors: [Aditi Thanekar](https://github.com/aditithanekar), [Amisha Prasad](https://github.com/amishap04), [Vaibhav Sharma](https://github.com/vas0090)

## Datasets Used: 
1. <a href= "https://www.kaggle.com/datasets/souvikrana17/indoor-plant-health-and-growth-dataset">Indoor Plant Health and Growth Database</a>
2. <a href= "https://www.kaggle.com/datasets/gabriellaamorim/gardening-q-and-a">Gardening Q&A </a>


## Files Needed for Our RAG
1. ```cse291a_phase_1.py```: This is our main file for implementing RAG, creating embeddings, feeding the embeddings into the Qdrant API, and calling the Triton LLM to answer our specific gardening related questions.
2. ```combined_plants_dataset.csv```: This is our main dataset - post-processing, which contains a combination of the 2 datasets mentioned above which are knowledge-based Q&A dataset and a quantitative plant dataset.



## User Guide: 
What you will need:
- An API Key from QDrant: Create a QDrant account: https://qdrant.tech/ and make a cluster --> you can call it cloud-based or using your local with docker, but we used cloud-based.
- Go to your AWS website, and set up an EC2 instance (AMI chosen was the AWS Linux Default)- we used a c6a.2xlarge with 8 CPUs and 16 GiB Memory, and added an additional 50GB of memory.
- Create a .env file with your API Key in this format: do not push the .env file to GitHub.
  ```
  QDRANT_URL = "https://yourqdranturlhere.gcp.cloud.qdrant.io" 
  QDRANT_API_KEY = "insert your QDrant API Key here"
  ```
- Ensure you have python3, pip3 and git installed and install these python libraries:

  ```
  pip3 install qdrant-client sentence-transformers pandas tqdm
  pip3 install numpy bs4 python-dotenv boto3
  ```
- Once you have an EC2 instance running, ensure you have AWS keys in order to run

  ```
  aws configure
  ```
  and paste the following: ```AWS_ACCESS_KEY_ID```, ```AWS_SECRET_ACCESS_KEY```, ```AWS_SESSION_TOKEN``` - when prompted, and us-west2 as the region.

Once your environment is set up: use the following command to run our code: 
```
python3 cse291a_phase_1.py
```




  
