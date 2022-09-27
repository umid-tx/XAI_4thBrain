# Problem
- While there has been breakthroughs in types of healthcare-specific AI modelling techniques, no significant attempt has been made in development of a machine learning model that provides a detailed description of why a categorical diagnosis of cancer is made based on the tissue images. 

- In this project, we attempt to develop a web application based on machine learning that will be capable of assessing whole-slide images of potentially cancerous tissues and produce a diagnosis and corresponding auto-generated explanation report. 

- More specifically, we intend to optimize the mitotic figure detection and count in WSIs to support pathologist with more confident/accurate decision making.


# Data
This project is based on the data used in the following paper:

- Aubreville, M., Bertram, C.A., Donovan, T.A. et al. A completely annotated whole slide image dataset of canine breast cancer to aid human breast cancer research. Sci Data 7, 417 (2020). https://doi.org/10.1038/s41597-020-00756-z

21 Whole Slide Image (WSI) of H&E-strained tissue dataset is available in figshare site: 
- Aubreville, M. et al. Dogs as model for human breast cancer - a completely annotated whole slide image dataset. figshare https://doi.org/10.6084/m9.figshare.c.4951281 (2020).

Annotated database associated with the dataset:

- https://github.com/DeepPathology/MITOS_WSI_CMC/tree/master/databases
- Definitions of each database: <br>
  - MEL - manually expert labelled <br>
  - ODAEL - object-detection augmented and expert labeled <br>
  - CODAEL - clustering- and object detection augmented manually expert labelled <br>


