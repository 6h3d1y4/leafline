# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information Base Data

### Dataset Source
- **Dataset Link:** [Training and Testing Data](https://sid.erda.dk/cgi-sid/ls.py?share_id=eFt21tspNe&current_dir=denmark&flags=f)

### Dataset Characteristics
- **Number of Observations:** 83 RGBI aerial images from Denmark. Each image contains a different number of trees. 
- **Number of Features:** [Total number of features in your dataset]

### Target Variable/Label
- **Label Name:** Trees
- **Label Type:** Instance Segmentation
- **Label Description:** Tree Counting and Crown Segmentation
- **Label Values:** Tree Count, Tree Location, Crown Segmentation, Crown Area, Tree Height
- **Label Distribution:** [Brief description of class balance for classification or value distribution for regression]

### Feature Description
- **Feature Group Images (red, gree, blue, infrared):** RGBI aerial images split into seperate channels, so that each pixel contains a value between 0 and 255. Data Type: png.   
- **Feature Group Additional calculated images (chm, NDVI):** additional channels for the RGBI images. chm contains the height above terrain (e.g. Buildings and Trees). NDVI stands for "normalized difference vegetation index", a common index for judging the vitatility of vegetation from remote sensing images. The index is calculated from the red and infrared channels. Values range from 0 to 1. Data Type: png.
- **Feature Group Labels (ann_kernel, annotation, boundary):** Contains labels/locations of tree instances and crown boundaries. Data Type: png and json.

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualization and insights
- Data quality assessment

## Dataset Information Finetuning and Prediction Data

### Dataset Source
- **Dataset Links:** 
    - [Prediction Images Kiel 20cm](https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/dl-dop20.html)
    - Finetuning and Testing Data Kiel: private
    - Prediction Images Kiel 7cm: private
- **Dataset Owner/Contact:** GDI@kiel.de; Landeshauptstadt Kiel, Amt für Bauordnung, Vermessung und Geoinformation 

### Notes
The Prediction Datasets are RGBI TrueOrthophoto tiles from 2025 (Format: GeoTIFF) with a resolution of 20cm and 7cm respectively. NDVI and height above terrain have been calculated.

The Finetuning Dataset is still being built. It will follow the format of the base training and testing dataset, but contain Images from Kiel.
The intention is to test how well a model trained on the opensource base dataset performs for the Kiel dataset and compare to a finetuned model.