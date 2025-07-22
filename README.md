# StructGAN Floorplan Shear Wall Prediction
This project provides code and a FastAPI service for predicting shear wall locations (red lines) on architectural floorplans using a trained pix2pix model (StructGAN).
Input floorplans are generated from .mat files and processed to PNG format before inference.

# Contents
pix.py – Model training code (U-Net generator and PatchGAN discriminator, using paired datasets)

mat_to_png_and_infer.py – Script to convert .mat files to PNG and run model inference

app.py – FastAPI API for inference from .mat file upload

outputs_pix2pix1/ – Folder with trained model (generator.pth)

outputs/ – Folder where predicted PNG outputs are saved

floorplan_pngs/ – Folder where .mat files are converted to PNG

pix.py/SimpleUNet – Model definition (imported as needed)

