# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection
- **Baseline Model Type:** UNet Model by Freudenberg et al. (2022). For more detail see Literature Review.
- **Rationale:** The project goal is to get accurate tree crown instance segmentation for Kiel city. Since there are already a lot of models, we chose this one, which has been pretrained on tree crown segmentation in Germany. As a baseline we evaluate how well the model perfoms as-is on the three image categories (summer, spring, high-res spring), to determine where finetuning is needed and which parameters (season, resolution) affect model performance in which way.

### Model Performance
- **Evaluation Metric:** TruePositive, FalsePositive, FalseNegative, Precision, Recall, F1-Score
- **Performance Score:**

| gebiet | aufloesung |	pred | gt | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|---:|
| BotGarten | 7.5cm | 16 | 357 | 0 | 16 |357 | 0.000 | 0.000 | 0.000 |
| BotGarten |20cm | 324	| 357 | 129 | 195 | 228 | 0.398 | 0.361 | 0.379 |
| BotGarten | 20cm-spring | 83 | 357 | 18 | 65 | 339 | 0.217 | 0.051 | 0.082 |
| HoernNord | 7.5cm	| 1 | 375 | 0 |	1 | 375 | 0.000 | 0.000 | 0.000 |
| HoernNord | 20cm | 139 | 375 | 74 | 65 | 301 | 0.532 | 0.197 | 0.288 |
| HoernNord	| 20cm-spring | 0 | 375 | 0 | 0 | 375 | 0.000 | 0.000 | 0.000 |
- **Cross-Validation Score:**

| metric | mean ± std |
|---|---:|
|pred | 93.83 ± 125.39 |
|gt | 366.00 ± 9.86 |
|tp | 36.83 ± 53.48 |
|fp | 57.00 ± 73.81 |
|fn | 329.17 ± 56.75 |
|precision | 0.19 ± 0.23 |
|recall | 0.10 ± 0.15 |
|f1 | 0.12 ± 0.17 |

### Evaluation Methodology
- **Data Split:** Train/Validation/Test split ratios: 70/15/15
- **Evaluation Metrics:**
    - IoU: This is not directly in the evaluation metric, but is used to map predicted trees to the ground truth instances. An IoU of 0.5 or greater allows the tree to be assigned to the ground truth instance. Each ground truth instance can only be mapped to once. A higher IoU makes the model more stringent, a lower one more forgiving.
    - TruePositive: Shows, how many trees were correctly identified.
    - FalsePositive: Shows, how many trees were predicted, that do not exist in reality.
    - FalseNegative: Shows, how many real trees were not predicted.
    - Precision: Shows the proportion of predicted trees that are actually real trees. It is calculated as TP / (TP + FP).
    - Recall: Shows the proportion of real trees that were successfully detected. It is calculated as TP / (TP + FN).
    - F1-Score: Combines precision and recall into a single metric by taking their harmonic mean. It is calculated as 2 × (Precision × Recall) / (Precision + Recall), with higher values indicating a better balance between correctly detecting trees and avoiding false detections.

### Metric Practical Relevance
The selected evaluation metrics provide complementary information about the quality of the tree detection model and its suitability for practical applications.

* **True Positives (TP)** indicate the number of trees that were correctly detected. A high TP count means that the model successfully identifies a large portion of the existing trees, making it suitable for tasks such as forest inventories or urban tree monitoring.

* **False Positives (FP)** represent predicted trees that do not exist in reality. In practice, a high FP count leads to an overestimation of the tree population, which may result in unnecessary field inspections, inaccurate vegetation statistics, or misguided management decisions.

* **False Negatives (FN)** represent real trees that the model failed to detect. A high FN count causes an underestimation of tree numbers and canopy coverage, reducing the reliability of applications such as ecological assessments, biomass estimation, or planning of maintenance activities.

* **Precision** measures the reliability of the model's predictions. A precision of 0.90, for example, means that 90% of all detected trees are correct, while 10% are false detections. High precision is particularly important when manual verification of predictions is expensive or time-consuming.

* **Recall** measures the model's ability to detect all existing trees. A recall of 0.90 indicates that 90% of all real trees were found, whereas 10% were missed. High recall is desirable when missing trees would negatively affect downstream analyses or decision-making.

* **F1-Score** combines precision and recall into a single measure by calculating their harmonic mean. It provides an overall assessment of detection performance and is especially useful when both false positives and false negatives are important. A high F1-score indicates that the model achieves a good balance between accurately identifying trees and minimizing missed detections.

For the task of tree crown detection, the F1-score is the most informative overall performance metric because it considers both the correctness of predictions and the completeness of tree detection. However, the individual values of precision and recall remain important, as they reveal whether performance is limited primarily by false detections or by missed trees, enabling more targeted model improvements.


## Next Steps
This baseline model serves as a reference point for evaluating finetuning of the model in the [Model Definition and Evaluation](../3_Model/README.md) phase.
To be useful for urban tree monitoring and management decisions the model needs improvement in all image categories. Especially for images from spring the model needs finetuning to be useful.
