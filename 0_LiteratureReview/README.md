# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1**: Wang et al. (2022): UNetFormer: A UNet-like Transformer for Efficient Semantic Segmentation of Remote Sensing Urban Scene Imagery

  - **[Link](https://arxiv.org/abs/2109.08937)**
  - **Objective**: Development of a new model structure for semantic segmentation of remote sensing images.
  - **Methods**: The proposed UNetFormer consists of a CNN-based encoder and a transformer-based decoder. The pre-trained ResNet18 is used as the encoder. The decoder uses three global-local transforer blocks and a feature refinement head to capture both local and global contexts without increasing the network-complexity unneccissarily. The model is then tested against multiple benchmark datasets and compared to other models. As evaluation metrics overall accuracy (OA), mean F1 score (F1), and mean intersection over union (mIoU) were used. The UAVid test set uses aerial images and eight semantic classes: Clutter, Building, Road, Static Car, Tree, Vegetation, Human, and Moving Car. The Vaihingen and Potsdam test sets use TrueOrthophotos and six semantic classes: Impervious surfaces, low vegetation, tree, car, building and background. The LoveDA test set uses TrueOrthophotos and seven semantic classes: Building, Road, Water, Barren, Forest, Agriculture, and Background.
  - **Outcomes**: The new UNetFormer runs faster and produces higher accuracy than other lightweight models for all test sets.
  - **Relation to the Project**: The code is Open-Source. We could use it and build on it, e.g. to include more datainputs like NIR-channel of TrueOrthophoto remote sensing images or height data. We could also test how well the model runs on datasets with different ground resolution or times of capture (e.g. winter vs. summer) and identify where training data or the model structure is lacking.

- **Source 2**: [Title of Source 2]

  - **[Link]()**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:

- **Source 3**: [Title of Source 3]

  - **[Link]()**
  - **Objective**:
  - **Methods**:
  - **Outcomes**:
  - **Relation to the Project**:
