import streamlit as st
from PIL import Image
from keras.utils import CustomObjectScope
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from PIL import Image

st.title('Multi-Label Genre Classification using Movie posters')

st.header('Upload a movie poster and get the genres it belongs to')

img_file_buffer = st.file_uploader("Upload a movie poster", type=["png", "jpg", "jpeg"])

def calculating_class_weights(y_true):
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

def weighted_loss(y_true, y_pred):
    return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)

y = pd.read_csv('data/train.csv').drop(['Id', 'Genre'],axis=1)
labels = list(y)

weights = calculating_class_weights(np.array(y))
with CustomObjectScope({'weighted_loss': weighted_loss}):
    model = load_model('model/model.h5')

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
    nor_image = image//255
    pil_image = Image.fromarray(nor_image)
    pil_image = pil_image.resize((224,224))

    st.image(
        np.array(image), caption=f"Processed image", use_column_width=True,
    )

    pred = model.predict(np.array(pil_image).reshape(1,224,224,3))
    most_likely = pred[0].argsort()[-2:][::-1]

    classes = []
    for i in range(len(most_likely)):
        classes.append(labels[i])
        
    st.text('  '.join(classes))
