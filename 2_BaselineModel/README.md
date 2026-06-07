# Baseline Model

**[Notebook](baseline_model.ipynb)**

## Baseline Model Results

### Model Selection

- **Baseline Model Type:** DeepForest pretrained tree crown detector.

- **Rationale:**  
  DeepForest was selected as the baseline because it provides a ready-to-use pretrained model for individual tree crown detection. It is applied zero-shot to the Denmark test dataset, meaning that no training or fine-tuning is performed on the target data. This makes it suitable as a simple lower-bound reference for later domain-specific models.

  The baseline intentionally uses only RGB bands because DeepForest expects standard RGB imagery as input. Although the Denmark dataset also contains infrared, NDVI, and CHM channels, these are excluded from the baseline so that later models can demonstrate the added value of richer input features and target-domain training.

### Model Performance

- **Evaluation Metric:** Precision, Recall, F1-score, TP, FP, FN, and mean matched circle IoU.

- **Performance Score:**

  | Metric | Score |
  |---|---:|
  | Precision | 0.0286 |
  | Recall | 0.0128 |
  | F1-score | 0.0177 |
  | True positives | 34 |
  | False positives | 1156 |
  | False negatives | 2613 |
  | Mean matched IoU | 0.3102 |

- **Cross-Validation Score:**  
  Cross-validation was not performed because this is a zero-shot pretrained baseline. The model was not trained on the Denmark dataset. It was evaluated directly on the provided Denmark test set.

### Evaluation Methodology

- **Data Split:**  
  No train/validation/test split was created for this baseline. The pretrained DeepForest model was evaluated directly on the Denmark test set containing 25 image tiles.

- **Evaluation Metrics:**  
  The model predictions and ground-truth annotations were converted into a common circle representation.

  DeepForest outputs bounding boxes. Each predicted bounding box was converted into a circle using the box centre as the circle centre and half of the box width as the radius.

  The ground-truth annotations are tree crown polygons. Each polygon was converted into an area-equivalent circle using the polygon centroid as the circle centre and `sqrt(area / pi)` as the radius.

  A greedy one-to-one matching strategy was used. A predicted circle was counted as a true positive if its circle IoU with an unmatched ground-truth circle was at least 0.5. Predictions without a match were counted as false positives, and unmatched ground-truth trees were counted as false negatives.

### Metric Practical Relevance

Precision indicates how many predicted tree crowns are actually correct. In a city-planning context, low precision means that the model may falsely mark non-tree objects as trees, which could distort estimates of existing tree coverage.

Recall indicates how many real tree crowns are detected. Low recall means that many trees are missed. This is especially important for urban canopy assessment because missed trees can lead to underestimation of tree coverage and poor downstream planning decisions.

The F1-score balances precision and recall. Since both false positives and missed trees matter for tree mapping, F1-score is a useful summary metric for comparing this baseline against later models.

Mean matched IoU measures the spatial overlap between matched predicted and ground-truth tree crowns. This provides additional information about localization quality for predictions that are considered correct.

The low baseline performance shows that a generic pretrained RGB-only tree detector does not transfer well to this Denmark dataset without fine-tuning. This establishes a meaningful lower-bound reference for the next model development phase.

## Next Steps

This baseline model serves as a reference point for evaluating more sophisticated models in the [Model Definition and Evaluation](../3_Model/README.md) phase.

The next model should aim to improve performance by using domain-specific training on the Denmark dataset and by incorporating the additional available channels, especially infrared, NDVI, and CHM.
