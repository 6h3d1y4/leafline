# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information Base Data

### Dataset Source
- **Dataset Link:** [Training and Testing Data](https://sid.erda.dk/cgi-sid/ls.py?share_id=eFt21tspNe&current_dir=denmark&flags=f)

### Dataset Characteristics
- **Number of Observations:** 183 aerial images from Denmark, split into 84 training, 25 test, and 74 validation samples. Each image is 1000 × 1014 px at 20 cm ground resolution.
- **Number of Features:** 6 input channels per image: Red, Green, Blue, Infrared, NDVI, CHM.

### Target Variable/Label
- **Label Name:** Trees
- **Label Type:** Instance Segmentation
- **Label Description:** Tree Counting and Crown Segmentation
- **Label Values:** Tree Count, Tree Location, Crown Segmentation, Crown Area, Tree Height
- **Label Distribution:** 51,740 annotated tree crowns in total. Mean of 282.7 trees per image, ranging from 0 (open fields) to 1,779 (dense forest). Crown areas are right-skewed: median 263 px² (~10.5 m², ~3.6 m diameter), mean 526 px², max 12,844 px².

### Feature Description
- **Feature Group Images (red, green, blue, infrared):** RGBI aerial images split into separate channels. The training split contains z-score normalised pixel values (approximately −5 to +6); the test and validation splits retain the original 0–255 range. Data Type: png.
- **Feature Group Additional calculated images (chm, NDVI):** Additional channels derived from the RGBI images. CHM (Canopy Height Model) contains height above terrain — strongly right-skewed with a large spike near zero (flat ground) and a tail up to ~45 (tall trees and buildings). NDVI (Normalized Difference Vegetation Index) is centred near zero due to normalisation, with a right tail representing dense vegetation. Data Type: png.
- **Feature Group Labels (ann_kernel, annotation, boundary):** Contains labels/locations of tree instances and crown boundaries. Annotation JSONs store each crown as a polygon under the key `"Trees"`. Data Type: png and json.

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:

- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis
- Data visualisation and insights
- Data quality assessment

### Key Findings

**Data Quality:** No missing files and no NaN pixels were detected across all 183 samples and 6 channels. The dataset is complete and requires no imputation.

**Feature Distributions:** The visible channels (RGB) are right-skewed and concentrated at low values. The infrared channel shows a bimodal distribution, with a second peak at higher values corresponding to vegetation — making it particularly informative for tree detection. NDVI and CHM distributions confirm that most pixels represent ground-level surfaces, with tree crowns occupying only a minority of each image.

**Feature Correlations:** RGB channels are nearly perfectly correlated with each other (r ≈ 1.00), offering largely redundant information at the image level. CHM is the strongest single predictor of tree count (r = 0.37), followed by NDVI (r = 0.22). RGB channels alone show near-zero correlation with tree count (r ≈ 0.00), justifying the use of all six input channels.

**Possible Biases:**
- *Geographic bias:* All images originate from Denmark. Species composition and urban structure may differ from the target region Kiel, Germany.
- *Seasonal bias:* Imagery was captured during the growing season (leaf-on). Performance on winter or autumn imagery may be reduced.
- *Resolution bias:* The dataset uses 20 cm resolution. The Kiel prediction data includes 7 cm tiles (DOP7), which will require fine-tuning.
- *Split bias:* The test split contains no images with zero tree density (min 39.9 trees/megapixel), while train and val include fully open scenes. The test set therefore does not evaluate performance on tree-free landscapes.
- *Domain shift (Kiel vs. Denmark):* Quantitative comparison is pending an NDA with Landeshauptstadt Kiel. Once data is available, a comparative distribution analysis will be added.

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