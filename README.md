# Overview:
- Data collection technologies like whole-slide imaging, 3D MRI, and X-Ray are all able to provide high-resolution data on biological tissue. 
- From this data, pathologists are able to perform diagnosis, teach and research.
- With the adoption of machine learning algorithms, there is a huge potential to automate and quantify detection of cancerous tissues. Even to surpass the current clinical approaches in terms of accuracy, reproducibility and objectivity. 

# Problem We Tried to Address:
- How to leverage deep learning technique to automate image analysis of potentially cancerous tissues?

# Deployment:
![image](https://user-images.githubusercontent.com/92175464/205825630-8cefb40a-ed37-4200-8df6-c1a363ccfeb9.jpeg)


# Data:
This project is based on the data used in the following paper:

- Aubreville, M., Bertram, C.A., Donovan, T.A. et al. A completely annotated whole slide image dataset of canine breast cancer to aid human breast cancer research. Sci Data 7, 417 (2020). https://doi.org/10.1038/s41597-020-00756-z

21 Whole Slide Image (WSI) of H&E-strained tissue dataset is available in figshare site: 
- Aubreville, M. et al. Dogs as model for human breast cancer - a completely annotated whole slide image dataset. figshare https://doi.org/10.6084/m9.figshare.c.4951281 (2020).

# Model:

Yolov5, https://github.com/ultralytics/yolov5, https://docs.ultralytics.com, Author: Jocher, G., Released: 18 May 2020

# Eigen-CAM:

Muhammad M.B., Yeasin, M., "Eigen-CAM: Class Activation Map using Principal Components", 1 Aug 2020, https://arxiv.org/abs/2008.00299'

https://jacobgil.github.io/pytorch-gradcam-book/EigenCAM%20for%20YOLO5.html


