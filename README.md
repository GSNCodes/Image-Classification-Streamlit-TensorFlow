# Image-Classification-Streamlit-Tensorflow
A basic web-app for image classification using Streamlit and Tensorflow

## Commands

To run the app locally, use the following command :-  
`streamlit run app.py`  

The webpage should open in the browser automatically.  
If it doesnt, the local URL would be output in the terminal, just copy it and open it in the browser manually.  
By default, it would be `http://localhost:8501/`


![output](misc/sample_output.png)

## Notes
* A simple flower classification model was trained using Tensorflow.  
* The weights are stored as `flower_model_trained.hdf5`.  
* The code to train the modify and train the model can be found in `model.py`.  
* The web-app created using Streamlit can be found in `app.py`


