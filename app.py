import streamlit as st
import os
import cv2 as cv
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import json
from io import StringIO
import tempfile

# Check if 'data' exists in session state, if not, initialize it
if 'data' not in st.session_state:
    st.session_state.data = []

# Check if 'data_aug_done' exists in session state, if not, initialize it
if 'data_aug_done' not in st.session_state:
    st.session_state.data_aug_done = False

# Check if 'numberof_layers' exists in session state, if not, initialize it
if 'numberof_layers' not in st.session_state:
    st.session_state.numberof_layers = None

if 'model_layers' not in st.session_state:
    st.session_state.model_layers = dict()

if 'data_aug' not in st.session_state:
    st.session_state.data_aug = False

if "model" not in st.session_state: 
    st.session_state.model = tf.keras.Sequential([])

if "optimizer" not in st.session_state:
    st.session_state.optimizer = "Adam"

if "loss" not in st.session_state:
    st.session_state.loss = "BinaryCrossentropy"

if "metrics" not in st.session_state:
    st.session_state.metrics = ["accuracy"]

if "batch" not in st.session_state:
    st.session_state.batch = 32

if "epochs" not in st.session_state:
    st.session_state.epochs = 10
    
if "validation_split" not in st.session_state:
    st.session_state.validation_split = 0.2

if "model_created" not in st.session_state: 
    st.session_state.model_created = False

# Creating dataset
df = pd.DataFrame(columns=["Images", "Label"])
if "dataset" not in st.session_state:
    st.session_state.dataset = df 

if "model_data" not in st.session_state:
    model_pram = dict()
    st.session_state.model_data = json.dumps(model_pram)

def augment_images(image, num_augmented):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # Normalize pixel values to the range [0, 1]
    image = np.array(image) / 255.0
    image = image.reshape((1,) + image.shape)

    augmented_images = []
    iterator = datagen.flow(image, batch_size=1)
    for _ in range(num_augmented):
        augmented_image = next(iterator)[0]
        augmented_images.append(augmented_image)

    return augmented_images

def get_image_dimensions(image):
    height, width = image.shape[:2]
    return width, height

def read_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    opencv_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return opencv_image

#model details 
def create_model():
    global width
    global height
    #model
    st.session_state.model = tf.keras.Sequential([])

    #saving the data 
    st.session_state.model_layers["Model"] = dict()
    for i in range(st.session_state.numberof_layers):
        value = st.session_state[f"model_layers_{i}"]
        st.session_state.model_layers["Model"][f"{value}_{i}"] = dict()
        if value == "CNN":
            st.session_state.model_layers["Model"][f"{value}_{i}"][f"Filter_{value}_{i}"] = st.session_state.get(f"Filter_{value}_{i}", None)
            st.session_state.model_layers["Model"][f"{value}_{i}"][f"Kernel_size_{value}_{i}"] = st.session_state.get(f"Kernel_size_{value}_{i}", None)
            st.session_state.model_layers["Model"][f"{value}_{i}"][f"ActivationL_{value}_{i}"] = st.session_state.get(f"ActivationL_{value}_{i}", None)
            if i == 0:
                st.session_state.model.add(tf.keras.layers.Rescaling(1./255, input_shape=(height, width, 3)))
                st.session_state.model.add(tf.keras.layers.Conv2D(filters= st.session_state.model_layers["Model"][f"{value}_{i}"][f"Filter_{value}_{i}"], 
                                             kernel_size=(st.session_state.model_layers["Model"][f"{value}_{i}"][f"Kernel_size_{value}_{i}"],st.session_state.model_layers["Model"][f"{value}_{i}"][f"Kernel_size_{value}_{i}"]),
                                              activation = st.session_state.model_layers["Model"][f"{value}_{i}"][f"ActivationL_{value}_{i}"]))
            else: 
                st.session_state.model.add(tf.keras.layers.Conv2D(filters= st.session_state.model_layers["Model"][f"{value}_{i}"][f"Filter_{value}_{i}"], 
                                             kernel_size=(st.session_state.model_layers["Model"][f"{value}_{i}"][f"Kernel_size_{value}_{i}"],st.session_state.model_layers["Model"][f"{value}_{i}"][f"Kernel_size_{value}_{i}"]),
                                              activation = st.session_state.model_layers["Model"][f"{value}_{i}"][f"ActivationL_{value}_{i}"]) ) 
        
        elif value in ["Max_filter", "Avg_filter"]:
            st.session_state.model_layers["Model"][f"{value}_{i}"][f"pool_size_{value}_{i}"] = st.session_state.get(f"pool_size_{value}_{i}", None)
            if value == "Max_filter":
                st.session_state.model.add(tf.keras.layers.MaxPool2D(pool_size=(st.session_state.model_layers["Model"][f"{value}_{i}"][f"pool_size_{value}_{i}"],st.session_state.model_layers["Model"][f"{value}_{i}"][f"pool_size_{value}_{i}"]))) 
            elif value == "Avg_filter":
                st.session_state.model.add(tf.keras.layers.AveragePooling2D(pool_size=(st.session_state.model_layers["Model"][f"{value}_{i}"][f"pool_size_{value}_{i}"],st.session_state.model_layers["Model"][f"{value}_{i}"][f"pool_size_{value}_{i}"])))
        
        elif value == "Dense":
            st.session_state.model_layers["Model"][f"{value}_{i}"][f"denselayer_{value}_{i}"] = st.session_state.get(f"denselayer_{value}_{i}", None)
            st.session_state.model_layers["Model"][f"{value}_{i}"][f"ActivationL_{value}_{i}"] = st.session_state.get(f"ActivationL_{value}_{i}", None)
            st.session_state.model.add(tf.keras.layers.Dense(st.session_state.model_layers["Model"][f"{value}_{i}"][f"denselayer_{value}_{i}"],activation =st.session_state.model_layers["Model"][f"{value}_{i}"][f"ActivationL_{value}_{i}"] ) )
        
        elif value == "Flatten":
            st.session_state.model.add(tf.keras.layers.Flatten())
    
    with StringIO() as s:
        st.session_state.model.summary(print_fn=lambda x: s.write(x + '\n'))
        summary_str = s.getvalue()
    
    st.session_state.model_created = True

    st.text(summary_str)

st.title("CNN Creator Application")
st.caption("This application allows users to easily create and download a CNN model from custom data with all possible optimizations for a CNN network")

# Streamlit data
label_inputs = dict()
Input_image = list()
st.subheader("Add Image")
data_aug_done = st.session_state.data_aug_done
img_data_1 = st.file_uploader("Add Your Image to Train", accept_multiple_files=True)

if img_data_1 is not None and len(img_data_1) > 0:
    for i, uploaded_file in enumerate(img_data_1):
        label = st.text_input(f"Label of {i}", key=f"label_{i}")
        st.image(uploaded_file)
        opencv_image = read_uploaded_image(uploaded_file)
        width, height = get_image_dimensions(opencv_image)
        label_inputs[i] = label
        Input_image.append(opencv_image)
    st.success("Image Uploaded Successfully")

    st.title("Data Augmentation")
    num_augmented = int(st.number_input("Number of Augmented Image", step=1))

    if st.button("Augment"):
        try:
            progress_bar = st.progress(0)
            total_iterations = len(Input_image) * num_augmented

            for idx, each in enumerate(Input_image):
                augmented_images = augment_images(each, num_augmented)
                for j, augmented_image in enumerate(augmented_images):
                    df.loc[len(df.index)] = [augmented_image, label_inputs[idx]]
                    progress_bar.progress((idx * num_augmented + j + 1) / total_iterations)

            st.session_state.data_aug_done = True
            st.session_state.data_aug = True
            st.session_state.data.append(st.session_state.data_aug_done)
            st.session_state.dataset = df
            st.success("Data augmentation done")
        except Exception as e:
            st.error(f"Data augmentation error: {str(e)}")


if st.session_state.data_aug:
    if st.session_state.data_aug_done:
        st.title("Model Parameters")
        numberof_layers = st.number_input("Number of layers", min_value=1, step=1)
        st.session_state.numberof_layers = numberof_layers

        for i in range(st.session_state.numberof_layers):
            layer = st.selectbox(f"Layer {i+1} type", ("CNN", "Dense", "Max_filter", "Avg_filter","Flatten"), key=f"model_layers_{i}")
            
            if layer == "CNN":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.number_input(f"Filter {i+1}", step=1, key=f"Filter_{layer}_{i}")
                with col2:
                    st.number_input(f"Kernel size {i+1}", step=1, key=f"Kernel_size_{layer}_{i}")
                with col3:
                    st.selectbox(f"Activation Layer {i+1}", ["relu", "selu", "softmax", "tanh", "sigmoid"], key=f"ActivationL_{layer}_{i}")
            
            elif layer in ["Max_filter", "Avg_filter"]:
                st.number_input(f"Pool size {i+1}", step=1, key=f"pool_size_{layer}_{i}")
            
            elif layer == "Dense":
                col1, col2 = st.columns(2)
                with col1:
                    st.number_input(f"Dense units {i+1}", step=1, key=f"denselayer_{layer}_{i}")
                with col2:
                    st.selectbox(f"Activation Layer {i+1}", ["relu", "selu", "softmax", "tanh", "sigmoid"], key=f"ActivationL_{layer}_{i}")

        if st.button("Create Model"):
            st.success("Model created")
            create_model()
            st.write(st.session_state.model_layers["Model"])

    if st.session_state.model_created:
        st.header("Model Training")
        st.session_state.optimizer = st.selectbox("Optimizer", ["Adam", "Adamax", "Sgd"])
        st.session_state.loss = st.selectbox("Loss", ["BinaryCrossentropy", "CategoricalCrossentropy", "MeanAbsoluteError", "MeanAbsolutePercentageError", "MeanSquaredError"])
        st.session_state.metrics = st.multiselect("Metrics", ["accuracy", "loss"])

        st.session_state.model.compile(optimizer=st.session_state.optimizer, loss=st.session_state.loss, metrics=st.session_state.metrics)

        st.session_state.batch = st.number_input("Batch size", 1, step=2)
        st.session_state.epochs = st.number_input("Epochs", min_value=1, step=2)
        st.session_state.validation_split = st.slider("Validation Split", 0.0, 1.0, step=0.01)

        try:
            X = np.array(st.session_state.dataset["Images"].tolist(), dtype=np.float32)
            y = np.array(st.session_state.dataset["Label"].tolist())

            # Convert string labels to integers
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

            # One-hot encode the labels for multi-class classification
            onehot_encoder = OneHotEncoder(sparse_output=False)
            y = onehot_encoder.fit_transform(y.reshape(-1, 1))

            st.session_state.model.fit(X, y, epochs=st.session_state.epochs, batch_size=st.session_state.batch, validation_split=st.session_state.validation_split)

            # Save model to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
            st.session_state.model.save(temp_file.name)
            with open(temp_file.name, "rb") as file: 
                st.download_button("Download Model", file.read(), file_name="model.h5")
            temp_file.close()
        except Exception as e:
            st.error(f"Model training error: {str(e)}")
