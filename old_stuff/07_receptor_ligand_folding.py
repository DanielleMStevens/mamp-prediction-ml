#-----------------------------------------------------------------------------------------------
# Coaker Lab - Plant Pathology Department UC Davis
# Author: Danielle M. Stevens
# Last Updated: 07/06/2020
# Script Purpose: 
# Inputs: 
# Outputs: 
#-----------------------------------------------------------------------------------------------

# import training file 
import pandas as pd
training_file = pd.read_excel("./All_LRR_PRR_ligand_data.xlsx", engine='openpyxl')


import torch
import esm

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()


training_file.iloc[3,7]

