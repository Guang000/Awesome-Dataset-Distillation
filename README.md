# Awesome-Dataset-Distillation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Papers-35-FF6F00" alt="PaperNum"/> ![Stars](https://img.shields.io/github/stars/Guang000/Awesome-Dataset-Distillation?color=yellow&label=Stars) ![Forks](https://img.shields.io/github/forks/Guang000/Awesome-Dataset-Distillation?color=green&label=Forks) ![Visitors](https://visitor-badge.glitch.me/badge?page_id=Guang000/Awesome-Dataset-Distillation) 

Awesome Dataset Distillation/Condensation Papers

Dataset distillation/condensation is the task of synthesizing a small dataset such that a model trained on the synthetic set will match the test accuracy of the model trained on the full dataset.

**This project is contributed and maintained by [Guang Li](https://www-lmd.ist.hokudai.ac.jp/member/guang-li/) and [Tongzhou Wang](https://www.tongzhouwang.info/).**

# Contents
- [Main](#main)
- [Privacy](#privacy)
- [Federated Learning](#fed)
- [Continual Learning](#continual)
- [Model Compression](#model)
- [Graph Neural Network](#gnn)
- [Neural Architecture Search](#nas)
- [Knowledge Distillation](#kd)
- [Medical](#medical)
- [Fashion](#fashion)
- [Reference](#ref)

:octocat: Code  🔥 Hot

<a name="main" />

# Main
+ [🔥Dataset Distillation](https://arxiv.org/abs/1811.10959) (Tongzhou Wang et al., 2018) **[[Project Page]](https://ssnl.github.io/dataset_distillation/)** [:octocat:](https://github.com/SsnL/dataset-distillation)
+ [Flexible Dataset Distillation: Learn Labels Instead of Images](https://arxiv.org/abs/2006.08572) (Ondrej Bohdal et al., NeurIPS2020 Workshop) [:octocat:](https://github.com/ondrejbohdal/label-distillation)
+ [Soft-Label Dataset Distillation and Text Dataset Distillation](https://arxiv.org/abs/1910.02551) (Ilia Sucholutsky et al., IJCNN2021) [:octocat:](https://github.com/ilia10000/dataset-distillation)
+ [🔥Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929) (Bo Zhao et al., ICLR2021) [:octocat:](https://github.com/VICO-UoE/DatasetCondensation)
+ [Dataset Meta-Learning from Kernel Ridge-Regression](https://arxiv.org/abs/2011.00050) (Timothy Nguyen et al., ICLR2021) [:octocat:](https://github.com/google/neural-tangents)
+ [Dataset Condensation with Differentiable Siamese Augmentation](https://arxiv.org/abs/2102.08259) (Bo Zhao et al., ICML2021)  [:octocat:](https://github.com/VICO-UoE/DatasetCondensation)
+ [Dataset Distillation with Infinitely Wide Convolutional Networks](https://arxiv.org/abs/2107.13034) (Timothy Nguyen et al., NeurIPS2021) [:octocat:](https://github.com/google/neural-tangents)
+ [Dataset Condensation with Distribution Matching](https://arxiv.org/abs/2110.04181) (Bo Zhao et al., 2021) [:octocat:](https://github.com/VICO-UoE/DatasetCondensation)
+ [🔥Dataset Distillation by Matching Training Trajectories](https://arxiv.org/abs/2203.11932) (George Cazenavette et al., CVPR2022) **[[Project Page]](https://georgecazenavette.github.io/mtt-distillation/)** [:octocat:](https://github.com/georgecazenavette/mtt-distillation)
+ [CAFE: Learning to Condense Dataset by Aligning Features](https://arxiv.org/abs/2203.01531) (Kai Wang et al., CVPR2022)  [:octocat:](https://github.com/kaiwang960112/cafe)
+ [Dataset Condensation with Contrastive Signals](https://arxiv.org/abs/2202.02916) (Saehyung Lee et al., ICML2022) [:octocat:](https://github.com/saehyung-lee/dcc)
+ [Dataset Condensation via Efficient Synthetic-Data Parameterization](https://arxiv.org/abs/2205.14959) (Jang-Hyun Kim et al., ICML2022) [:octocat:](https://github.com/snu-mllab/efficient-dataset-condensation)
+ [Synthesizing Informative Training Samples with GAN](https://arxiv.org/abs/2204.07513) (Bo Zhao et al., 2022) [:octocat:](https://github.com/vico-uoe/it-gan)
+ [Dataset Distillation using Neural Feature Regression](https://arxiv.org/abs/2206.00719) (Yongchao Zhou et al., 2022)
+ [DC-BENCH: Dataset Condensation Benchmark](https://arxiv.org/abs/2207.09639) (Justin Cui et al., 2022) [:octocat:](https://github.com/justincui03/dc_benchmark)

<a name="privacy" />

# Privacy
+ [Soft-Label Anonymous Gastric X-ray Image Distillation](https://arxiv.org/abs/2104.02857) (Guang Li et al., ICIP2020) [:octocat:](https://github.com/Guang000/Awesome-Dataset-Distillation)
+ [SecDD: Efficient and Secure Method for Remotely Training Neural Networks](https://arxiv.org/abs/2009.09155) (Ilia Sucholutsky et al., AAAI2021 Student Abstract)
+ [🔥Privacy for Free: How does Dataset Condensation Help Privacy?](https://arxiv.org/abs/2206.00240) (Tian Dong et al., ICML2022, **[Outstanding Paper Award](https://icml.cc/virtual/2022/awards_detail))** 

<a name="fed" />

# Federated Learning
+ [Federated Learning via Synthetic Data](https://arxiv.org/abs/2008.04489) (Jack Goetz et al., 2020)
+ [Distilled One-Shot Federated Learning](https://arxiv.org/abs/2009.07999) (Yanlin Zhou et al., 2020)
+ [FedSynth: Gradient Compression via Synthetic Data in Federated Learning](https://arxiv.org/abs/2204.01273) (Shengyuan Hu et al., 2022)
+ [FedDM: Iterative Distribution Matching for Communication-Efficient Federated Learning](https://arxiv.org/abs/2207.09653) (Yuanhao Xiong et al., 2022)

<a name="continual" />

# Continual Learning
+ [Reducing Catastrophic Forgetting with Learning on Synthetic Data](https://arxiv.org/abs/2004.14046) (Wojciech Masarczyk et al., CVPR2020 Workshop)
+ [Condensed Composite Memory Continual Learning](https://arxiv.org/abs/2102.09890) (Felix Wiewel et al., IJCNN2021) [:octocat:](https://github.com/FelixWiewel/CCMCL)
+ [Distilled Replay: Overcoming Forgetting through Synthetic Samples](https://arxiv.org/abs/2103.15851) (Andrea Rosasco, 2021) [:octocat:](https://github.com/andrearosasco/DistilledReplay)
+ [Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks](https://arxiv.org/abs/2206.02916) (Zhiwei Deng et al., 2022)
+ [Sample Condensation in Online Continual Learning](https://arxiv.org/abs/2206.11849) (Mattia Sangermano et al., IJCNN2022)

<a name="model" />

# Model Compression
+ [Compressed Gastric Image Generation Based on Soft-Label Dataset Distillation for Medical Data Sharing](https://www.journals.elsevier.com/computer-methods-and-programs-in-biomedicine) (Guang Li et al., CMPB2022)
+ [PRANC: Pseudo RAndom Networks for Compacting deep models](https://arxiv.org/abs/2206.08464) (Parsa Nooralinejad et al., 2022) [:octocat:](https://github.com/UCDvision/PRANC)

<a name="gnn" />

# Graph Neural Network
+ [Graph Condensation for Graph Neural Networks](https://arxiv.org/abs/2110.07580) (Wei Jin et al., ICLR2022) [:octocat:](https://github.com/chandlerbang/gcond)
+ [Condensing Graphs via One-Step Gradient Matching](https://arxiv.org/abs/2206.07746) (Wei Jin et al., KDD2022) [:octocat:](https://github.com/ChandlerBang/GCond/tree/main/KDD22_DosCond)
+ [Graph Condensation via Receptive Field Distribution Matching](https://arxiv.org/abs/2206.13697) (Mengyang Liu et al., 2022)

<a name="nas" />

# Neural Architecture Search
+ [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://arxiv.org/abs/1912.07768) (Felipe Petroski Such et al., ICML2020) [:octocat:](https://github.com/uber-research/GTN)

<a name="kd" />

# Knowledge Distillation
+ [Knowledge Condensation Distillation](https://arxiv.org/abs/2207.05409) (Chenxin Li et al., ECCV2022) [:octocat:](https://github.com/dzy3/KCD)

<a name="medical" />

# Medical
+ [Soft-Label Anonymous Gastric X-ray Image Distillation](https://arxiv.org/abs/2104.02857) (Guang Li et al., ICIP2020) [:octocat:](https://github.com/Guang000/Awesome-Dataset-Distillation)
+ [Compressed Gastric Image Generation Based on Soft-Label Dataset Distillation for Medical Data Sharing](https://www.journals.elsevier.com/computer-methods-and-programs-in-biomedicine) (Guang Li et al., CMPB2022)

<a name="fashion" />

# Fashion
+ [Wearable ImageNet: Synthesizing Tileable Textures via Dataset Distillation](https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/html/Cazenavette_Wearable_ImageNet_Synthesizing_Tileable_Textures_via_Dataset_Distillation_CVPRW_2022_paper.html) (George Cazenavette et al., CVPR2022 Workshop) **[[Project Page]](https://georgecazenavette.github.io/mtt-distillation/)** [:octocat:](https://github.com/georgecazenavette/mtt-distillation)

<a name="ref" />

# Reference

**If you find some papers useful for your research, please cite these papers. [[Reference]](https://github.com/Guang000/Awesome-Dataset-Distillation/blob/main/REFERENCE.md)**
