## Stylised-Controllable-Image-Captioning-Streamlit App
This is a project that generates stylised captions for images whose length can be controlled 
externally.

### Prerequisites
The model is based on Pytorch and Transformers. For this app you need 
Streamlit.

### Project Structure
This project has four major parts :
1. `inference.py` - This contains code for the Machine Learning model to predict captions based on the image and controllable 
parameter.
2. `app.py` - This contains Streamlit app that receives an image through GUI and computes the required captions.
3. `src` - This folder contains the model and other utilities required to generate captions.
4. `data` - This folder contains JSON vocab map along with utilities functions to use them.
5. `ckpts` - This folder will contain the required model and `config.json` file.

### Running the project
Run app.py using below command to start the Streamlit app.
```
streamlit run app.py
```
This would create a serialized version of our model into a file model.pkl

By default, flask will run on port 8501.

2. Navigate to URL http://localhost:8501

3. Proceed to upload the image and wait for results.