![leafline logo](https://github.com/6h3d1y4/leafline/blob/main/CoverImage/leafline_logo_inverted.png)

# Leafline: Tree-Cover Mapping for Kiel

## Repository Link

[https://github.com/6h3d1y4/leafline]

## Description

Leafline is a geospatial AI project for mapping urban tree cover in Kiel, Germany, using high-resolution RGBI orthomosaic imagery and deep-learning-based instance segmentation.

The project seeks to assess the effectiveness of using deep-learning to identify tree crowns in Kiel city. 
The city currently has extensive manually collected data on trees on city-owned grounds. This leaves out data about trees on any private ground, leaving a large data-gap when trying to assess green spaces in the city. 
High-resolution aerial images are usually collected by plane once a year and theoretically present a great chance to fill this data-gap. Filling this gap would allow the city to better track changes in tree numbers and tree cover, allowing for comprehensive analysis of city climate, shade corridors, green islands, and greenery development. 
Identifying individual trees combined with high-resolution aerial data further allows for tree species identification, as well as individual tree crown area, height, and health assessments.

Using pre-trained tree-segmentation models however comes with major challenges and uncertainties: 
1. Most models are trained on summer images, while the city mostly collects images in spring. 
2. Since publicly available data, which models are generally trained on, is lower-resolution, it is unclear what impact the higher resolution images have on model performance.
3. Height data is shown to benefit crown-segmentation, but is rarely included in models.

This project aims to adress these uncertainties, by finetuning an existing tree-segmentation model and testing it on different data sets. We also expand the model to include height data. We chose an existing tree segmentation model, since the task is quite complex and there are already a lot of models available. 
Our findings also give insight into a possible workflow to adjust the model to other geographical areas, since it is more efficient and requires less data than to train a model from scratch. This is very helpful for further development regarding tree sementation, because data availability in particular is one of the main challenges in aerial image segmentation.

### Task Type

Image Classification - Instance Segmentation

### Results Summary

#### Best Model Performance
- **Best Model:** [Name and type of the best-performing model"]
- **Evaluation Metric:** [Primary metric used, e.g., Accuracy, F1-Score, MSE, MAE]
- **Final Performance:** [Best score achieved, e.g., 95% accuracy, F1-score of 0.87, MSE of 0.12]

#### Model Comparison
- **Baseline Performance:** [Baseline model performance for comparison]
- **Improvement Over Baseline:** [Quantitative improvement, e.g., "+12% accuracy", "25% reduction in MSE"]
- **Best Alternative Model:** [Second-best model and its performance]

#### Key Insights
- **Most Important Features:** [Top 3-5 features that drive model performance]
- **Model Strengths:** [What the model does well]
- **Model Limitations:** [Known limitations and failure cases]
- **Business Impact:** [Practical implications of the model performance]

## Documentation

1. **[Literature Review](0_LiteratureReview/README.md)**
2. **[Dataset Characteristics](1_DatasetCharacteristics/exploratory_data_analysis.ipynb)**
3. **[Baseline Model](2_BaselineModel/baseline_model.ipynb)**
4. **[Model Definition and Evaluation](3_Model/model_definition_evaluation)**
5. **[Presentation](4_Presentation/README.md)**

## Cover Image

