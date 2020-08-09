## Stylised-Controllable-Image-Captioning-Streamlit App
This is a project that generates stylised captions for images whose length can be controlled 
externally.

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

### Running the project
Run app.py using below command to start the Streamlit app.
```
streamlit run app.py
```
This would create a serialized version of our model into a file model.pkl

By default, Streamlit will run on port 8501.

2. Navigate to URL http://localhost:8501

3. Proceed to upload the image and wait for results.