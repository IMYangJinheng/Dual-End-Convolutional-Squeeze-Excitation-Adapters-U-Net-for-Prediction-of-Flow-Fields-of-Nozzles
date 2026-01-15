# Dual-End-Convolutional-Squeeze-Excitation-Adapters-U-Net-for-Prediction-of-Flow-Fields-of-Nozzles
This repository contains the implementation of the paper: **"Incremental Learning for Flow Field Reconstruction in Variable-Geometry Nozzles"**(Under Review).

We propose a DE-ConvSE Adapter U-Net for rapid nozzle flow prediction, in whicha U-Net backbone is augmented with Dual-End Convolutional Squeeze-and-Excitation adapters. 
The network integrates geometric topology encoding and boundary condition mapping to accurately reconstruct velocity, pressure, andtemperature distributions. 
To enable efficient adaptation to previously unseen nozzle geometries, an adapter-basedincremental learning strategy is introduced, where the pretrained backbone is frozen and only a small set of adapter parameters is updated. 
This strategy substantially improves the generalization capability of the model while maintaining high computational efficiency and mitigating catastrophic forgetting. Validation on a convergent–divergent nozzle benchmark demonstrates that the proposed approach achieves high reconstruction accuracy, 
with maximum symmetric relative errors below 9.14% and mean absolute errors below 5.21%. Compared with CFD solvers, the proposed method is mesh-free and provides orders-of-magnitude acceleration, highlighting its potential for real-time flow prediction and
design optimization in advanced fluid dynamic systems.

![Overview](asset/FIG_1.png)

## 🚀 Key Features

* **DE-ConvSE Adapter U-Net:** A U-Net backbone augmented with Dual-End Convolutional Squeeze-and-Excitation adapters.
* **Incremental Learning:** Freezes the backbone and updates only the adapters to adapt to new geometries efficiently.
* **Geometry-Aware Inputs:** Utilizes Signed Distance Fields (SDF) and Identifier Matrices (IM) to encode boundary conditions.
* **High Performance:**
    * **Speedup:** >300x faster than traditional CFD solvers.
    * **Accuracy:** Max Symmetric Relative Error (SRE) < 9.14%, and Mean Absolute Errors (MAE) below 5.21%

## 🛠️ Model Architecture

![Model Architecture_1](asset/FIG_5/.png)

![Model Architecture_2](asset/FIG_6.png)

## 📂 Dataset Preparation
The nozzle configuration developed by the Jet Propulsion Laboratory (JPL) is adopted as the reference geometry.

![Base Nozzle_1](assets/architecture.png)

![Base_Nozzle_2](assets/architecture.png)

The dataset consists of nozzle geometries parameterized by Bézier curves and their corresponding CFD flow fields (Temperature, Velocity, Pressure).

* **Inputs:** Signed Distance Field (SDF) and Identifier Matrix (IM).

![Input_1](assets/architecture.png)

![Input_2](assets/architecture.png)
* **Outputs:** Flow field images/tensors.
