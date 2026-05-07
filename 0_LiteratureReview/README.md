# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: Wang et al. (2022): UNetFormer: A UNet-like Transformer for Efficient Semantic Segmentation of Remote Sensing Urban Scene Imagery

  - **[Link](https://arxiv.org/abs/2109.08937)**
  - **Objective**: Development of a new model structure for semantic segmentation of remote sensing images.
  - **Methods**: The proposed UNetFormer consists of a CNN-based encoder and a transformer-based decoder. The pre-trained ResNet18 is used as the encoder. The decoder uses three global-local transforer blocks and a feature refinement head to capture both local and global contexts without increasing the network-complexity unneccissarily. The model is then tested against multiple benchmark datasets and compared to other models. As evaluation metrics overall accuracy (OA), mean F1 score (F1), and mean intersection over union (mIoU) were used. The UAVid test set uses aerial images and eight semantic classes: Clutter, Building, Road, Static Car, Tree, Vegetation, Human, and Moving Car. The Vaihingen and Potsdam test sets use TrueOrthophotos and six semantic classes: Impervious surfaces, low vegetation, tree, car, building and background. The LoveDA test set uses TrueOrthophotos and seven semantic classes: Building, Road, Water, Barren, Forest, Agriculture, and Background.
  - **Outcomes**: The new UNetFormer runs faster and produces higher accuracy than other lightweight models for all test sets.
  - **Relation to the Project**: The code is Open-Source. We could use it and build on it, e.g. to include more datainputs like NIR-channel of TrueOrthophoto remote sensing images or height data. We could also test how well the model runs on datasets with different ground resolution or times of capture (e.g. winter vs. summer) and identify where training data or the model structure is lacking.

---

* **Source 2: Freudenberg et al. (2022): Individual tree crown delineation in high-resolution remote sensing images based on U-Net**
  * [Link](https://link.springer.com/article/10.1007/s00521-022-07640-4)
  * **Objective:** Automatic delineation of individual tree crowns in aerial and satellite imagery without requiring 3D height data (e.g. LiDAR).
  * **Methods:** U-Net based deep learning framework trained on 2D optical images only. Tested on 30 cm WorldView-3 satellite imagery (urban area, India) and 5 cm aerial imagery (forested area, Germany). Evaluation via IoU for the tree cover mask and polygon-level accuracy and recall for individual crown delineation. The method produces irregular polygons rather than bounding boxes and also provides a tree cover mask for areas where individual crowns are not separable.
  * **Outcomes:** IoU of 71.2% for satellite imagery and 81.9% for aerial images on the tree cover mask. The model is trainable with small amounts of annotated data and requires no LiDAR or 3D height information.
  * **Relation to the Project:** Directly transferable to our use case. The aerial imagery resolution is comparable to DOP20, the U-Net architecture is compatible with our planned model stack, and the Germany-based test case demonstrates generalisability to Central European vegetation. The low annotation requirement is relevant given our limited labelled Kiel data.

---

* **Source 3: Rottensteiner et al. (2012): The ISPRS Benchmark on Urban Object Classification and 3D Building Reconstruction**
  * [Link](https://isprs-annals.copernicus.org/articles/I-3/293/2012/)
  * **Objective:** Provision of a standardised benchmark dataset for urban object detection from aerial imagery to enable comparable evaluation of different algorithms.
  * **Methods:** Dataset consisting of airborne imagery and laser scanner data covering urban areas in Potsdam and Vaihingen. Researchers submitted results for urban object detection and 3D building reconstruction, evaluated against reference data. Classes include buildings, trees, low vegetation, impervious surfaces, cars, and background. Standard evaluation metrics are overall accuracy (OA), mean F1 score, and mean IoU.
  * **Outcomes:** Systematic comparison and analysis of submitted methods to identify promising strategies for automatic urban object extraction from airborne sensor data, as well as common failure modes of state-of-the-art approaches at the time.
  * **Relation to the Project:** This is our primary pretraining dataset. The paper defines the class taxonomy, evaluation metrics, and scientific context that we adopt directly for our project. Understanding the benchmark setup is essential for correctly interpreting pretraining results and for justifying our choice of baseline classes before fine-tuning on Kiel DOP20 data.
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:
