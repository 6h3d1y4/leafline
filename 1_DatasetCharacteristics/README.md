# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information Base Data

### Dataset Source
- **Dataset Links:** 
    - [Images Kiel 20 cm](https://geodaten.schleswig-holstein.de/gaialight-sh/_apps/dladownload/dl-dop20.html)
    - Images Kiel 7.5 cm: private
- **Dataset Owner/Contact:** GDI@kiel.de; Landeshauptstadt Kiel, Amt für Bauordnung, Vermessung und Geoinformation 

### Dataset Characteristics
- **Number of Observations:** 9 aerial images of different sizes, ranging from a size of covering roughly 1.300 m² to around 51.800 m². The images are RGBI TrueOrthophotos. Each image is available in 3 configurations: (1) summer 20 cm resolution, (2) spring 20 cm resolution, (3) spring 7.5 cm resolution. The images are split into 4 training, 3 validation, and 2 testing images. For both resolutions height data (height above terrain) is also available.
- **Number of Features:** 6 input channels per image: Red, Green, Blue, Infrared, NDVI, CHM (height). Each of the 9 images has a corresponding shapefile with labeled tree crowns. 

### Target Variable/Label
- **Label Name:** Trees
- **Label Type:** Instance Segmentation
- **Label Description:** Tree Counting and Crown Segmentation
- **Label Values:** Tree Count, Tree Location, Crown Segmentation, Crown Area
- **Label Distribution:** 3178 annotated tree crowns over 9 images. Median crown area is 29.68 m².

### Feature Description
- **Feature Group Images (red, green, blue, infrared):** RGBI aerial images split into separate channels. All values range from 0 to 255 and need to be normalized during preprocessing. Data Type: GeoTIFF.
- **Feature Group Additional calculated images (chm, NDVI):** Additional channels derived from the RGBI images. CHM (Canopy Height Model) contains height above terrain — strongly right-skewed with a small amount negative (terrain below sea level), a large spike near zero (flat ground), and a tail up to ~45 (tall trees and buildings). NDVI (Normalized Difference Vegetation Index) is calculated from red and infrared channels on the fly during preprocessing and is centred near zero due to normalisation, with a right tail representing dense vegetation. Data Type: GeoTIFF.
- **Feature Group Labels (tree labels):** Polygons showing individual tree crown areas. Manually annotated for the project from aerial images, measured tree locations, and google street view. These need to be converted to pixel masks during preprocessing. Data Type: shapefile.

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

**Feature Distributions:** The visible channels (RGB) from the summer images are relativly evenly distributed with slight spikes at each end. The NIR channel is heavily right skewed, resulting from the strong signal of healthy vegetation in summer. The RGB and NIR channels from the spring images are left skewed, showing the different reflection behaviour before summer vegetation. CHM distributions show a strong peak at around zero, owning to the flat terrain of northern germany. The negative values represent terrain below sea level. Overall the distribution is right-skewed due to the long tail of high values representing tree crowns and buildings. 

**Feature Correlations:** 
RGBI channels strongly correlate with each other and slightly with height and crown area, independent of seasonality. While the correlation between NIR and crown area is not that strong in summer, it is notably weaker in spring, being almost absent altogether. Height is most strongly correlated with crown area, indicating that including height data in the model has good potential to increase model performance.


**Possible Biases:**
- *Regional Bias*: Ground-Truth data is only collected within the city of Kiel. This will likely limit the performance in rural contexts, other climate zones with differnt trees, and mountainous regions.
- *Seasonal Bias*: Most models are trained on summer (leaf-on) imagery, since it allows for much better tree identification and segmentation. Since aerial images collected in Kiel city are generally collected in spring (leaf-off), a model will require finetuning to accuratly capture trees in sping images. The dataset contains these images.
- *Resolution Bias*: Most models are trained on resolutions around 20cm, since public data is not available at higher resolution. This dataset contains high-resolution (7.5cm) images from spring 2025.
- This dataset contains images from multiple seasons and with multiple resolutions and is therefore well suited to analyze the impact of seasonal and resolution bias on model output.
- *Tree sampling bias*: Large trees appear more often than smaller trees and are better represented in the height data. This might cause small trees to be underrepresented in the model and less accuratly predicted.

