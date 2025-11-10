"""
This extract the features from the whole volume with pyradiomic

__author__ = "Lariza Sandoval"
__license__ = ""
__email__ = ""
__year__ = "2025"
"""

import os
import numpy as np
import pandas as pd
from radiomics import featureextractor
from NiftyIO import readNifty
import re
from radiomics import setVerbosity
setVerbosity(60) 
from featureExtractionFunctions import *
   

# ------------------------------------
# PARAMETERS
# ------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INPUT_ROOT = os.path.join(BASE_DIR, "..", "data", "input")
DATA_OUTPUT_ROOT = os.path.join(BASE_DIR, "..", "data", "output")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
IMAGE_DIR = os.path.join(DATA_INPUT_ROOT, "image")
MASK_DIR  = os.path.join(DATA_INPUT_ROOT, "nodule_mask") # change to the best mask from segmentation pipeline 
RESULTS_CSV = os.path.join(RESULTS_DIR, "features.csv")

#List of all images and mask 
image_Names= os.listdir(IMAGE_DIR)
mask_Names= os.listdir(MASK_DIR)
mask_Names_Dict = {elemento.replace('best_mask_',''): elemento.replace('best_mask_','') for elemento in mask_Names}

#Reading meta data Excel
df_meta_data = pd.read_excel( os.path.join(DATA_INPUT_ROOT,'MetadatabyNoduleMaxVoting.xlsx'), 
                      sheet_name='ML4PM_MetadatabyNoduleMaxVoting', 
                      engine='openpyxl'
        )


# Use a parameter file, this customizes the extraction settings and
# also specifies the input image types to use and
# which features should be extracted.
params = os.path.join(DATA_INPUT_ROOT,'FeaturesExtraction_Params.yaml')

# Initializing the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(params)
nodules_features = []

#Extracting features from all whole images
for image in image_Names:
        
    image_Name= os.path.join(IMAGE_DIR,image)
    mask_Name=os.path.join(MASK_DIR,mask_Names_Dict[image])
    image_Data, meta1 = readNifty(image_Name, CoordinateOrder='xyz')
    mask_Data, meta2 = readNifty(mask_Name, CoordinateOrder='xyz')
    

    patient_id= image[0:14]
    nodule_id= image[17:19] if image[17:19].isdigit() else image[17]
    
    print(f"-------Paciente: {patient_id}, Nodule: {nodule_id}-----------")

    ### -----PREPROCESSING-----
    #image_Data = ShiftValues(image_Data, value=1024)
    image_Data = SetRange(image_Data, in_min=0, in_max=4000)
    #image = SetGrayLevel(image, levels=24)

    diagnosis = df_meta_data[(df_meta_data.patient_id.astype(str)==patient_id)  & 
                      (df_meta_data.nodule_id.astype(str)==nodule_id)
                    ].Diagnosis_value.values[0]
    
      
    # Get back to the format sitk
    img_sitk = sitk.GetImageFromArray(image_Data)
    mask_sitk = sitk.GetImageFromArray(mask_Data)
                  
    # Recover the pixel dimension in X , Y and Z
    (x1, y1, z1) = meta1.spacing
    (x2, y2, z2) = meta2.spacing
    img_sitk.SetSpacing((float(x1), float(y1),float(z1)))
    mask_sitk.SetSpacing((float(x2), float(y2),float(z2)))
    

    # Extract features
    featureVector = extractor.execute(img_sitk, mask_sitk, voxelBased=False) 
    nodule = GetFeatures(featureVector, 0, patient_id, nodule_id, diagnosis)
    nodules_features.append(nodule)
    df = pd.DataFrame.from_dict(nodules_features)
 
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
df.to_csv(RESULTS_CSV, index=False)
print("---Extraction Pipeline finish---")
