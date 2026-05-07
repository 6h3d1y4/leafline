![leafline logo](https://github.com/6h3d1y4/leafline/blob/main/CoverImage/leafline_logo_full.png)
# Leafline: Tree-Cover Mapping for Kiel

## Repository Link

[https://github.com/6h3d1y4/leafline]

## Description

Leafline is a geospatial AI project for mapping urban tree cover in Kiel, Germany, using high-resolution RGBI orthomosaic imagery, remote sensing data, and deep-learning-based semantic segmentation.

The project aims to develop a reproducible workflow to detect and quantify tree canopy cover across the city. A segmentation model is pretrained on existing aerial imagery dataset and fine-tuned using manually annotated Kiel RGBI orthomosaic tiles. The final model will generate georeferenced tree-cover masks that can be used to calculate canopy percentage.

The broader goal is to support urban climate resilience and green infrastructure planning by identifying areas with limited tree cover. Future extensions include overlaying tree-cover maps with vulnerable-population locations such as schools, kindergartens, hospitals, clinics, and elderly-care homes to assess local canopy availability and prioritize greening interventions.

### Task Type

[Image Classification]

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

![Project Cover Image](CoverImage/cover_image.png)
