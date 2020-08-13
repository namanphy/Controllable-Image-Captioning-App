<div style="align:left">

# Stylised Controllable Image Captioning App

[![release](https://img.shields.io/badge/current%20release-v0.1-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)](https://github.com/namanphy/stylised-controllable-image-captioning-StreamlitApp/blob/master/LICENSE)

</div>

## Introduction
Vanilla Image captioning model usually aims for factual precision but lack of engagement. Such goal limits the application 
of image captioning. We tried to build an image captioning model that sounds like human and relevant to the image. Such 
features are valuable in many applications such as social media and advertisement. 
To increase the usability of the model, a control is added on different attributes, such as **length of sentence** and 
**control of emojis**.

### Prerequisites
The model is based on Pytorch and Transformers. For this app you need 
Streamlit.
Download the following files and put them into `ckpts` folder:  
- [model checkpoint](https://drive.google.com/file/d/1ciN8Iz1qE1JTmo8TEMMUxBN-LvUovd6B/view?usp=sharing)
- [word map file](https://drive.google.com/file/d/1MLRazOJwn52dfYnPP83u0qsKj3uf6Jn6/view?usp=sharing)

### Project Structure
This project has four major parts :
1. `app.py` - This contains Streamlit app that receives an image through GUI and computes the required captions.
2. `src` - This folder contains the model and other utilities required to generate captions.
3. `ckpts` - This folder will contain the required model and `config.json` file.


## Get it run quickly

Clone the repo and navigate to the repo directory.
```
git clone https://github.com/namanphy/stylised-controllable-image-captioning-StreamlitApp.git
cd stylised-controllable-image-captioning-StreamlitApp
```

You can quickly build and run the docker image locally with:
```
docker build -t ctrl-img-cap-streamlit:latest .
docker run -d -p 8501:8501 ctrl-img-cap-streamlit:latest
```
And navigate to URL **http://localhost:8501** for the app. *(Streamlit runs on port 8501 by default)*

#### Installation
**Alternatively for development you can create and manage a python environment** : 

Make a python3.6 environment *(either pipenv or conda env)*, install requirements.
```
conda create -n 'app-env' python=3.6
pip install -r requirements.txt
```

And, run `app.py` using below command to start the Streamlit app.
```
streamlit run app.py
```

"https://download.pytorch.org/models/resnet101-5d3b4d8f.pth" to /root/.cache/torch/checkpoints/resnet101-5d3b4d8f.pth

#### Running Unit Test
```
pytest -s tests
```
