import pandas as pd
#import plotly.express as px
#import plotly.figure_factory as ff
import streamlit as st
#import streamlit.components.v1 as components
#import math
import numpy as np 
#import SlideRunner.general.dependencies
#from SlideRunner.dataAccess.database import Database
#from SlideRunner.dataAccess.annotations import *
#import os
import openslide
#import sys
#from pathlib import Path
import pandas as pd
#import random
import matplotlib.pyplot as plt
from openslide import open_slide
#from os import system

import wsi_tile
from wsi_tile import * 

import eigencam
from eigencam import *

import cv2
import torch
from PIL import Image 
Image.MAX_IMAGE_PIXELS = 1000000000

st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
<style>
.tabs {
background-image: #F5F5F5;
}
</style>
""",
    unsafe_allow_html=True,
)
# Display a sample Whole Slide Image as a logo.
st.image("460906c0b1fe17ea5354.svs.png", use_column_width='always')

st.markdown("<h1 style='text-align: center; color: black;'>Mitotic Figure Detection App</h1>", unsafe_allow_html=True)

# Create tabs for separation of tasks
tab1, tab2, tab3, tab4 = st.tabs([" 🌱 About the App ", " 🗃 Import Whole Slide Image ", " 🔎 Model Detection Results ", " 🤓 Model Explainability "])

with tab1:
    st.subheader('App Description')
    st.markdown('This is a web application based on machine \
         learning that is capable of assessing a whole-slide image of potentially \
         cancerous tissues and detecting mitotic/non-mitotic figures.')

    st.subheader('How to use this App')
    st.markdown(
        """
        To use this application:
        - Upload your SVS file under ***Import SVS file*** tab.
        - Under ***Model Detection Results*** tab, select your desired input and view your results.
        - Based on the model prediction, check ***Model Explainability*** tab where an Eigen-CAM method is used to highlight features important for model prediction.
        """
    )

    st.subheader('References')
    st.markdown(
        """
        This project is based on the data used in the following paper:
        * Aubreville, M., Bertram, C.A., Donovan, T.A. et al. A completely annotated whole slide image dataset of canine breast cancer to aid human breast cancer research. Sci Data 7, 417 (2020). https://doi.org/10.1038/s41597-020-00756-z
        
        21 Whole Slide Images (WSIs) of H&E-strained tissue dataset used in model training are available in figshare site:
        * Aubreville, M. et al. Dogs as model for human breast cancer - a completely annotated whole slide image dataset. figshare https://doi.org/10.6084/m9.figshare.c.4951281 (2020).

        Model reference:
        * https://github.com/ultralytics/yolov5%7D%7D

        """
    )

with tab2:    
    # Section Header
    st.subheader("Import your Whole Slide Image for mitotic figure detection")

    uploaded_file = st.file_uploader('Select your SVS file', type=['svs'])
    
    # caching the uploaded file
    @st.experimental_memo
    def svs_file():
        uploaded_svs = uploaded_file
        return uploaded_svs 
    
    # Wait to execute until file is uploaded
    if uploaded_file is not None:  
        input_svs = svs_file()
    else:
        st.stop()
    
    slide = open_slide(input_svs)

    # caching thumbnail image 
    @st.experimental_memo
    def get_thumb_image():
        slide_thumb_600 = slide.get_thumbnail(size=(600,600))
        return slide_thumb_600

    st.image(get_thumb_image(), caption='Thumbnail image of your SVS file', use_column_width='always')

    st.markdown("""Dimensions (in pixels) of the input SVS is : {}""".format(slide.dimensions), unsafe_allow_html=True)


with tab3:
    var_container = st.container()
    fig_container = st.container()
    pred_container = st.container()
    output_container = st.container()

    # Container for input variables to used in extracting tiles and inferencing 
    with var_container:

        # Section Header
        st.subheader("Select input variables")

        sample_options = st.selectbox('Select sample number of images for detection:', options = [5, 10, 50, 100, 'All'])

        Conf = st.selectbox('Select a confidence score:', options = [0.4, 0.5, 0.6, 0.7, 0.8])

        IoU = st.selectbox('Select Intersection of Union (IoU) threshold:', options = [0.5, 0.6, 0.7, 0.8])

        if sample_options == 'All':
            sample_options = 'None'
        
        # tiled_images, cols_rows_loc = get_tiles(slide, patchSize = 128, sample_images = sample_options)

        # n_rows = int(sample_options/5)

        # fig, axs = plt.subplots(n_rows, 5) #, figsize=(10,n_rows*2))
        # axs = axs.flatten()
        # for i in range(len(tiled_images)):
        #     axs[i].imshow(tiled_images[i])
        #     axs[i].title.set_text("Tile #{}".format(i+1))
        #     axs[i].axis('off')
        # fig.savefig('sample_tile_images.png')
    
    # with fig_container:

    #     st.subheader("Sample Tiles")
    #     st.image("sample_tile_images.png", use_column_width='always')
    
    # Container for model prediction 
    with pred_container:
            st.subheader("Model Prediction")
 
            mitotic_images, mitotic_pandas = detect_mitosis(slide, patchSize = 128, sample_images = sample_options, Conf=Conf, IoU=IoU)

            display_df = mitotic_pandas[['label', 'confidence']]
            x_coord_mitosis = get_mitosis_box_location(mitotic_pandas, patchSize=128)['mitosis_coord_x_min'] + 25 # approximate location of mitotic figure in x-direction
            y_coord_mitosis = get_mitosis_box_location(mitotic_pandas, patchSize=128)['mitosis_coord_y_min'] + 25 # approximate location of mitotic figure in y-direction 
            # approximate because assuming the bounding box size is 50, thus putting half of that to get the ~ center of mitotic figure
            display_df['X-coordinate'], display_df['Y-coordinate'] = x_coord_mitosis, y_coord_mitosis
            display_df = display_df.rename(columns={'label': 'Tile Label', 'confidence': 'Confidence Score'})
            st.caption('List of tiles with mitotic figure, model detection confidence score, approximate coordinates on Whole Slide Image')
            st.text('Note: Origin (0,0) for image is at top left corner')

            st.dataframe(display_df)

            # n_rows2 = math.ceil(len(mitotic_pandas)/2)

            # fig2, axs2 = plt.subplots(n_rows2, 2, figsize=(10,n_rows*2))
            # axs2 = axs2.flatten()
            # for i in range(len(mitotic_images)):
            #     axs2[i].imshow(draw_box(mitotic_images[i], mitotic_pandas.iloc[i]))
            #     axs2[i].title.set_text("Tile #{}".format(display_df.iloc[i][0]))
            #     axs2[i].axis('off')
            # fig2.savefig('detected_tile_images.png')

            # st.image("detected_tile_images.png", use_column_width='always')

            # st.image(mitotic_images[0], use_column_width='always')
            counter = 0
            for i in range(len(mitotic_images)):
                c1,mid,c2 = st.columns([2,0.5,2])
                if counter < len(mitotic_images):
                    c1.image(draw_box(mitotic_images[counter], mitotic_pandas.iloc[counter]), caption=display_df.iloc[counter][0], use_column_width='always')
                    counter +=1
                    if counter > (len(mitotic_images)-1):
                        break
                    else:
                        c2.image(draw_box(mitotic_images[counter], mitotic_pandas.iloc[counter]), caption=display_df.iloc[counter][0], use_column_width='always')
                counter += 1


    with output_container:
        st.subheader("Visualization of Detected Mitotic Figures on Whole Slide Image")

        st.image(draw_global_box(slide, mitotic_pandas, patchSize=128, scf=0.02), use_column_width='always')


with tab4:
    eigen_container = st.container()
    eigen_output_container = st.container()

    with eigen_output_container:

        st.subheader('Eigen-CAM')
        st.markdown(
            """
            - An attempt is made at understanding what features are triggering the model to make the object detection via the use of ***Eigen-CAM*** approach (CAM stands for class activation map).
            - Eigen-CAM computes and visualizes the principle components of the learned features/representations from the convolutional layers.
            - According to the researchers who developed this approach (see reference below): "Eigen-CAM was found to be robust against classification errors made \
            by fully connected layers in CNNs, does not rely on the backpropagation of gradients, class relevance score, maximum activation \
            locations, or any other form of weighting features. In addition, it works with all CNN models without the need to modify layers or retrain models."
            """)
        st.text('Reference: Muhammad M.B., Yeasin, M., "Eigen-CAM: Class Activation Map using Principal Components", 1 Aug 2020, https://arxiv.org/abs/2008.00299')
    
    with eigen_output_container:

        st.subheader('Eigen-CAM Output')
        for i in range(len(mitotic_images)):
            xai_image, orig_image = get_eigen_cam(img_nparray=mitotic_images[i], Conf=Conf, IoU=IoU)
            hstacked_img = Image.fromarray(np.hstack((orig_image, xai_image)))
            st.image(hstacked_img, caption=display_df.iloc[i][0], use_column_width='always')    


    