import streamlit as st
import os
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

# Check if 'data' exists in session state, if not, initialize it
if 'data' not in st.session_state:
    st.session_state.data = []

# Check if 'data_aug_done' exists in session state, if not, initialize it
if 'data_aug_done' not in st.session_state:
    st.session_state.data_aug_done = False

# Check if 'numberof_layers' exists in session state, if not, initialize it
if 'numberof_layers' not in st.session_state:
    st.session_state.numberof_layers = None
    

# Creating dataset
df = pd.DataFrame(columns=["Images", "Label"])


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
    for _ in range(num_augmented):
        augmented_image = datagen.flow(image, batch_size=1).next()[0]
        augmented_images.append(augmented_image)

    return augmented_images


def read_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    opencv_image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return opencv_image


st.title("CNN Creator Application")
st.caption("This application is a basic application which will provide the user to easily create and download CNN or convolutional model from their custom data with all the optimizations possible for a CNN network")

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
        label_inputs[i] = label
        Input_image.append(opencv_image)

    st.success("Image Uploaded Successfully")

    st.title("Data Augmentation")
    num_augmented = int(st.number_input("Number of Augmented Image",step=1))
    if st.button("Augment"):
        # Display a progress bar
        try:
            progress_bar = st.progress(0)
            for percent_complete in range(1, num_augmented + 1):
                for idx, each in enumerate(Input_image):
                    augmented_images = augment_images(each, 1)
                    progress_bar.progress((percent_complete / num_augmented))

                    for j in range(len(augmented_images)):
                        df.loc[len(df.index)] = [augmented_images[j], label_inputs[idx]]
            st.success("Data augmentation done")
            st.session_state.data_aug_done = True  # Update session state variable
            st.session_state.data.append(st.session_state.data_aug_done)  # Append to data list

        except:
            st.error("Data augmentation error")
            print(df)


if st.session_state.data_aug_done:
    st.title("Model Descriptions")
    numberof_layers = st.number_input("Number of layers", min_value=1, step=1)
    st.session_state.numberof_layers = numberof_layers  # Update session state variable
    st.session_state.data.append(st.session_state.numberof_layers)  # Append to data list

    for i in range(st.session_state.numberof_layers):
        st.selectbox("",("CNN","Flatten","Max_Filte","Avg_filter"),key=i)
