# Learning of Geometry Estimation for Wrinkled Garments

My semester project at ETH Zurich.

This task is to reconstruct the geometry of a wrinkled garment/deformed cloth using a single-view depth simulated image.

The idea of using Graph Neural Network to model the deformed cloth is inspired by MeshGraphNets: https://arxiv.org/pdf/2010.03409.pdf

The idea of using template-based deformation to model the geometry estimation of wrinkled garments is inspired by https://arxiv.org/abs/1905.03244.

Part of the code is adapted from the pytorch implementation of meshgraphnets from https://github.com/wwMark/meshgraphnets.

## Paper/Report
You can find it here: https://github.com/ligengen/Deformed-cloth-reconstruction/blob/main/Semester_Project_Report.pdf

## Method overview
<img src="img/overview.png" width="50%" height="50%" />

## Deformation (message passing) visualization
<img src="img/mp.png" width="50%" height="50%" />

## Qualitative visualization
<img src="img/qual.png" width="50%" height="50%" />

## Data augmentation for occlusion
<img src="img/occ.png" width="50%" height="50%" />


## Code for training a new model:
```
python main.py
```

## Code for test a model:
```
python main.py --phase=test --pt_file=./experiments/exp_288759_resnet34_edge_no_image_feature_attention/bestmodel_0189_0.0082321205.pt

```
### Modify parameters according to your needs
