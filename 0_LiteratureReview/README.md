# Literature Review

Approaches or solutions that have been tried before on similar projects.

**Summary of Each Work**:

- **Source 1: Freudenberg et al. (2022): Individual tree crown delineation in high-resolution remote sensing images based on U-Net**
  
  - [Link](https://link.springer.com/article/10.1007/s00521-022-07640-4)
  - **Objective:** Automatic delineation of individual tree crowns in aerial and satellite imagery without requiring 3D height data (e.g. LiDAR).
  - **Methods:** U-Net based deep learning framework trained on 2D optical images only. Tested on 30 cm WorldView-3 satellite imagery (urban area, India) and 5 cm aerial imagery (forested area, Germany). Evaluation via IoU for the tree cover mask and polygon-level accuracy and recall for individual crown delineation. The method produces irregular polygons rather than bounding boxes and also provides a tree cover mask for areas where individual crowns are not separable.
  - **Outcomes:** IoU of 71.2% for satellite imagery and 81.9% for aerial images on the tree cover mask. The model is trainable with small amounts of annotated data and requires no LiDAR or 3D height information.
  - **Relation to the Project:** Directly transferable to our use case. The aerial imagery resolution is comparable to DOP20, the U-Net architecture is compatible with our planned model stack, and the Germany-based test case demonstrates generalisability to Central European vegetation. The low annotation requirement is relevant given our limited labelled Kiel data.

---

- **Source 2: Khan et al. (2025): DeepTrees: Tree Crown Segmentation and Analysis in Remote Sensing Imagery with PyTorch**

  - [Link](https://doi.org/10.21105/joss.08056)
  - **Objective**: The paper presents DeepTrees, an open-source Python package for tree crown segmentation and analysis in remote sensing imagery. The software is designed to support scalable deep learning workflows for individual tree crown delineation from high-resolution satellite, aerial, and UAV imagery. The stated objective is to provide a reproducible and extensible framework that combines segmentation, model training, fine-tuning, and ecological analysis within a single geospatial workflow.
  - **Methods**: DeepTrees is implemented in PyTorch and PyTorch Lightning and provides modular components for data loading, preprocessing, model training, inference, and evaluation. The package supports training models from scratch, fine-tuning pre-trained U-Net models, and integrating alternative backbone architectures, including geospatial foundation models. The package includes pre-trained models and a labelled tree crown dataset for the Halle region.
  - **Outcomes**: The primary outcome is an open-source software framework that unifies tree crown segmentation, model development, active learning, and ecological analysis. DeepTrees produces multiple geospatial outputs, including tree masks, crown outlines, polygons, uncertainty maps, and allometric metrics, enabling efficient integration into ecological monitoring and forest management workflows. The publication focuses on describing the capabilities and design of the software rather than presenting a quantitative benchmark or comparative performance evaluation. Therefore, no experimental performance results or accuracy comparisons are reported in the paper itself.
  - **Relation to the Project**: The software is highly relevant to the project because it directly addresses AI-based individual tree crown delineation from high-resolution aerial imagery. In particular, DeepTrees supports fine-tuning of existing models, incorporates pre-trained networks, and is designed for adaptation to new datasets and imaging conditions through transfer learning. These capabilities align closely with the project's investigation of whether existing models require fine-tuning for high-quality urban tree crown detection in Kiel.
  The package also supports multiple data sources, geospatial processing workflows, and integration of additional model architectures, making it suitable for evaluating different aerial image characteristics, as well as adding height data. 

---

- **Source 3: Rottensteiner et al. (2012): The ISPRS Benchmark on Urban Object Classification and 3D Building Reconstruction**

  - [Link](https://isprs-annals.copernicus.org/articles/I-3/293/2012/)
  - **Objective:** Provision of a standardised benchmark dataset for urban object detection from aerial imagery to enable comparable evaluation of different algorithms.
  - **Methods:** Dataset consisting of airborne imagery and laser scanner data covering urban areas in Potsdam and Vaihingen. Researchers submitted results for urban object detection and 3D building reconstruction, evaluated against reference data. Classes include buildings, trees, low vegetation, impervious surfaces, cars, and background. Standard evaluation metrics are overall accuracy (OA), mean F1 score, and mean IoU.
  - **Outcomes:** Systematic comparison and analysis of submitted methods to identify promising strategies for automatic urban object extraction from airborne sensor data, as well as common failure modes of state-of-the-art approaches at the time.
  - **Relation to the Project:** This is our primary pretraining dataset. The paper defines the class taxonomy, evaluation metrics, and scientific context that we adopt directly for our project. Understanding the benchmark setup is essential for correctly interpreting pretraining results and for justifying our choice of baseline classes before fine-tuning on Kiel data.

---

- **Source 4: Teng et al. (2025):**

  - [Link](https://arxiv.org/abs/2506.04970)
  - **Objective:** The paper investigates the suitability of the Segment Anything Model (SAM) for automatic individual tree crown instance segmentation in high-resolution drone imagery. It further evaluates whether integrating Digital Surface Model (DSM) elevation data and fine-tuning SAM can improve segmentation performance across three forest settings: boreal plantations, temperate forests, and tropical forests.
  - **Methods:** The authors compare multiple SAM-based segmentation approaches with a custom Mask R-CNN baseline for tree crown instance segmentation. In addition to evaluating SAM without task-specific training, they propose BalSAM, a model that integrates DSM elevation information with SAM. DSMs are derived from the same RGB drone imagery through photogrammetry, allowing the incorporation of structural height information without requiring additional data acquisition. The methods are evaluated on datasets representing three distinct forest ecosystems.
  - **Outcomes:** The study finds that out-of-the-box SAM models do not outperform a custom Mask R-CNN, even when carefully designed prompts are used. However, end-to-end fine-tuning of SAM and the integration of DSM elevation data both improve segmentation performance. The proposed BalSAM model demonstrates particular promise in plantation settings, suggesting that height information can provide complementary structural cues for tree crown delineation. The paper concludes that adapting large pre-trained vision models to the task and incorporating elevation data are promising directions for future tree crown segmentation research.
  - **Relation to the Project:** The paper is highly relevant to the project, as both focus on AI-based individual tree crown segmentation from aerial imagery. Its investigation of DSM integration directly relates to the project's objective of evaluating the influence of height data on model performance. Furthermore, the comparison between out-of-the-box and fine-tuned SAM models aligns closely with the project's research on the necessity of fine-tuning existing foundation models. While the paper uses high-resolution drone imagery in forest environments rather than aerial imagery of an urban area, its findings provide valuable insights into the potential benefits and limitations of foundation models and elevation data for tree crown delineation.