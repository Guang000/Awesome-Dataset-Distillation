# Awesome-Dataset-Distillation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
<img src="https://img.shields.io/badge/Contributions-Welcome-278ea5" alt="Contrib"/> <img src="https://img.shields.io/badge/Number%20of%20Papers-95-FF6F00" alt="PaperNum"/> ![Stars](https://img.shields.io/github/stars/Guang000/Awesome-Dataset-Distillation?color=yellow&label=Stars) ![Forks](https://img.shields.io/github/forks/Guang000/Awesome-Dataset-Distillation?color=green&label=Forks)

A curated list of awesome papers on dataset distillation and related applications, inspired by [awesome-computer-vision](https://github.com/jbhuang0604/awesome-computer-vision).

**Dataset distillation** is the task of synthesizing a small dataset such that models trained on it achieve high performance on the original large dataset. A dataset distillation algorithm takes as **input** a large real dataset to be distilled (training set), and **outputs** a small synthetic distilled dataset, which is evaluated via testing models trained on this distilled dataset on a separate real dataset (validation/test set). A good small distilled dataset is not only useful in dataset understanding, but has various applications (e.g., continual learning, privacy, neural architecture search, etc.). This task was first introduced in the 2018 paper [*Dataset Distillation* [Tongzhou Wang et al., '18]](https://www.tongzhouwang.info/dataset_distillation/), along with a proposed algorithm using backpropagation through optimization steps.

In recent years (2019-now), dataset distillation has gained increasing attention in the research community, across many institutes and labs. More papers are now being published each year. These wonderful researches have been constantly improving dataset distillation and exploring its various variants and applications.

**This project is curated and maintained by [Guang Li](https://www-lmd.ist.hokudai.ac.jp/member/guang-li/), [Bo Zhao](https://www.bozhao.me/), and [Tongzhou Wang](https://www.tongzhouwang.info/).**

#### [How to submit a pull request?](./CONTRIBUTING.md)

+ :globe_with_meridians: Project Page
+ :octocat: Code
+ :book: `bibtex`

## Citing Awesome-Dataset-Distillation

If you find this project useful for your research, please use the following BibTeX entry.

```
@misc{li2022awesome,
  author={Li, Guang and Zhao, Bo and Wang, Tongzhou},
  title={Awesome-Dataset-Distillation},
  howpublished={\url{https://github.com/Guang000/Awesome-Dataset-Distillation}},
  year={2022}
}
```

## Contents
- [Main](#main)
  - [Early Work](#early-work)
  - [Gradient/Trajectory Matching Surrogate Objective](#gradient-objective)
  - [Distribution/Feature Matching Surrogate Objective](#feature-objective)
  - [Better Optimization](#optimization)
  - [Distilled Dataset Parametrization](#parametrization)
  - [Generative Prior](#generative)
  - [Label Distillation](#label)
  - [Dataset Quantization](#quant)
  - [Multimodal Distillation](#multi)
  - [Benchmark](#benchmark)
  - [Survey](#survey)
- [Applications](#applications)
  - [Continual Learning](#continual)
  - [Privacy](#privacy)
  - [Medical](#medical)
  - [Federated Learning](#fed)
  - [Graph Neural Network](#gnn)
  - [Neural Architecture Search](#nas)
  - [Fashion, Art, and Design](#fashion)
  - [Knowledge Distillation](#kd)
  - [Recommender Systems](#rec)
  - [Blackbox Optimization](#blackbox)
  - [Trustworthy](#trustworthy)
  - [Retrieval](#retrieval)
  - [Text](#text)
  - [Tabular](#tabular)

[Media Coverage](#media)<br/>
[Acknowledgments](#ack)

<a name="main" />

## Main
+ [Dataset Distillation](https://arxiv.org/abs/1811.10959) (Tongzhou Wang et al., 2018) [:globe_with_meridians:](https://ssnl.github.io/dataset_distillation/) [:octocat:](https://github.com/SsnL/dataset-distillation) [:book:](./citations/wang2018datasetdistillation.txt)

<a name="early-work" />

### Early Work
+ [Gradient-Based Hyperparameter Optimization Through Reversible Learning](https://arxiv.org/abs/1502.03492) (Dougal Maclaurin et al., ICML 2015) [:octocat:](https://github.com/HIPS/hypergrad) [:book:](./citations/maclaurin2015gradient.txt)

<a name="gradient-objective" />

### Gradient/Trajectory Matching Surrogate Objective
+ [Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929) (Bo Zhao et al., ICLR 2021) [:octocat:](https://github.com/VICO-UoE/DatasetCondensation) [:book:](./citations/zhao2021datasetcondensation.txt)
+ [Dataset Condensation with Differentiable Siamese Augmentation](https://arxiv.org/abs/2102.08259) (Bo Zhao et al., ICML 2021)  [:octocat:](https://github.com/VICO-UoE/DatasetCondensation) [:book:](./citations/zhao2021differentiatble.txt)
+ [Dataset Distillation by Matching Training Trajectories](https://arxiv.org/abs/2203.11932) (George Cazenavette et al., CVPR 2022) [:globe_with_meridians:](https://georgecazenavette.github.io/mtt-distillation/) [:octocat:](https://github.com/georgecazenavette/mtt-distillation) [:book:](./citations/cazenavette2022dataset.txt)
+ [Dataset Condensation with Contrastive Signals](https://arxiv.org/abs/2202.02916) (Saehyung Lee et al., ICML 2022) [:octocat:](https://github.com/saehyung-lee/dcc) [:book:](./citations/lee2022dataset.txt)
+ [Delving into Effective Gradient Matching for Dataset Condensation](https://arxiv.org/abs/2208.00311) (Zixuan Jiang et al., 2022) [:book:](./citations/jiang2022delving.txt)
+ [Loss-Curvature Matching for Dataset Selection and Condensation](https://arxiv.org/abs/2303.04449) (Seungjae Shin & Heesun Bae et al., AISTATS 2023) [:octocat:](https://github.com/SJShin-AI/LCMat) [:book:](./citations/shin2023lcmat.txt)
+ [Minimizing the Accumulated Trajectory Error to Improve Dataset Distillation](https://arxiv.org/abs/2211.11004) (Jiawei Du & Yidi Jiang et al., CVPR 2023) [:octocat:](https://github.com/AngusDujw/FTD-distillation) [:book:](./citations/du2023minimizing.txt)
+ [Scaling Up Dataset Distillation to ImageNet-1K with Constant Memory](https://arxiv.org/abs/2211.10586) (Justin Cui et al., ICML 2023) [:book:](./citations/cui2022scaling.txt)
+ [DREAM: Efficient Dataset Distillation by Representative Matching](https://arxiv.org/abs/2302.14416) (Yanqing Liu & Jianyang Gu et al., ICCV 2023) [:octocat:](https://github.com/lyq312318224/DREAM) [:book:](./citations/liu2023dream.txt)
+ [Towards Lossless Dataset Distillation via Difficulty-Aligned Trajectory Matching](https://arxiv.org/abs/2310.05773) (Ziyao Guo et al., 2023) [:globe_with_meridians:](https://gzyaftermath.github.io/DATM/) [:octocat:](https://github.com/GzyAftermath/DATM) [:book:](./citations/guo2023datm.txt)

<a name="feature-objective" />

### Distribution/Feature Matching Surrogate Objective
+ [CAFE: Learning to Condense Dataset by Aligning Features](https://arxiv.org/abs/2203.01531) (Kai Wang & Bo Zhao et al., CVPR 2022) [:octocat:](https://github.com/kaiwang960112/cafe) [:book:](./citations/wang2022cafe.txt)
+ [Dataset Condensation with Distribution Matching](https://arxiv.org/abs/2110.04181) (Bo Zhao et al., WACV 2023) [:octocat:](https://github.com/VICO-UoE/DatasetCondensation) [:book:](./citations/zhao2023distribution.txt)
+ [Improved Distribution Matching for Dataset Condensation](https://arxiv.org/abs/2307.09742) (Ganlong Zhao et al., CVPR 2023) [:octocat:](https://github.com/uitrbn/IDM) [:book:](./citations/zhao2023idm.txt)
+ [DataDAM: Efficient Dataset Distillation with Attention Matching](https://arxiv.org/abs/2310.00093) (Ahmad Sajedi & Samir Khaki, ICCV 2023) [:octocat:](https://github.com/DataDistillation/DataDAM) [:book:](./citations/sajedi2023datadam.txt)

<a name="optimization" />

### Better Optimization
+ [Optimizing Millions of Hyperparameters by Implicit Differentiation](https://arxiv.org/abs/1911.02590) (Jonathan Lorraine et al., AISTATS 2020) [:octocat:](https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/Optimizing_Millions_of_Hyperparameters_by_Implicit_Differentiation) [:book:](./citations/lorraine2020optimizing.txt) 
+ [Dataset Meta-Learning from Kernel Ridge-Regression](https://arxiv.org/abs/2011.00050) (Timothy Nguyen et al., ICLR 2021) [:octocat:](https://github.com/google/neural-tangents) [:book:](./citations/nguyen2021kip.txt)
+ [Dataset Distillation with Infinitely Wide Convolutional Networks](https://arxiv.org/abs/2107.13034) (Timothy Nguyen et al., NeurIPS 2021) [:octocat:](https://github.com/google/neural-tangents) [:book:](./citations/nguyen2021kipimprovedresults.txt)
+ [On Implicit Bias in Overparameterized Bilevel Optimization](https://proceedings.mlr.press/v162/vicol22a.html) (Paul Vicol et al., ICML 2022) [:book:](./citations/vicol2022implicit.txt)
+ [Dataset Distillation using Neural Feature Regression](https://arxiv.org/abs/2206.00719) (Yongchao Zhou et al., NeurIPS 2022) [:globe_with_meridians:](https://sites.google.com/view/frepo) [:octocat:](https://github.com/yongchao97/FRePo) [:book:](./citations/zhou2022dataset.txt)
+ [Efficient Dataset Distillation using Random Feature Approximation](https://arxiv.org/abs/2210.12067) (Noel Loo et al., NeurIPS 2022) [:octocat:](https://github.com/yolky/RFAD) [:book:](./citations/loo2022efficient.txt)
+ [Accelerating Dataset Distillation via Model Augmentation](https://arxiv.org/abs/2212.06152) (Lei Zhang & Jie Zhang et al., CVPR 2023) [:octocat:](https://github.com/ncsu-dk-lab/Acc-DD) [:book:](./citations/zhang2023accelerating.txt)
+ [Dataset Distillation with Convexified Implicit Gradients](https://arxiv.org/abs/2302.06755) (Noel Loo et al., ICML 2023) [:octocat:](https://github.com/yolky/RCIG) [:book:](./citations/loo2023dataset.txt)
+ [On the Size and Approximation Error of Distilled Sets](https://arxiv.org/abs/2305.14113) (Alaa Maalouf & Murad Tukan, NeurIPS 2023) [:book:](./citations/maalouf2023size.txt)
+ [Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective](https://arxiv.org/abs/2306.13092) (Zeyuan Yin & Zhiqiang Shen et al., NeurIPS 2023) [:globe_with_meridians:](https://zeyuanyin.github.io/projects/SRe2L/) [:octocat:](https://github.com/VILA-Lab/SRe2L) [:book:](./citations/yin2023sre2l.txt)

<a name="parametrization" />

### Distilled Dataset Parametrization
+ [Dataset Condensation via Efficient Synthetic-Data Parameterization](https://arxiv.org/abs/2205.14959) (Jang-Hyun Kim et al., ICML 2022) [:octocat:](https://github.com/snu-mllab/efficient-dataset-condensation) [:book:](./citations/kim2022dataset.txt)
+ [Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks](https://arxiv.org/abs/2206.02916) (Zhiwei Deng et al., NeurIPS 2022) [:octocat:](https://github.com/princetonvisualai/RememberThePast-DatasetDistillation) [:book:](./citations/deng2022remember.txt)
+ [On Divergence Measures for Bayesian Pseudocoresets](https://arxiv.org/abs/2210.06205) (Balhae Kim et al., NeurIPS 2022) [:octocat:](https://github.com/balhaekim/bpc-divergences) [:book:](./citations/kim2022divergence.txt)
+ [Dataset Distillation via Factorization](https://arxiv.org/abs/2210.16774) (Songhua Liu et al., NeurIPS 2022) [:octocat:](https://github.com/Huage001/DatasetFactorization) [:book:](./citations/liu2022dataset.txt)
+ [Synthesizing Informative Training Samples with GAN](https://arxiv.org/abs/2204.07513) (Bo Zhao et al., NeurIPS 2022 Workshop) [:octocat:](https://github.com/vico-uoe/it-gan) [:book:](./citations/zhao2022synthesizing.txt)
+ [PRANC: Pseudo RAndom Networks for Compacting Deep Models](https://arxiv.org/abs/2206.08464) (Parsa Nooralinejad et al., 2022) [:octocat:](https://github.com/UCDvision/PRANC) [:book:](./citations/nooralinejad2022pranc.txt)
+ [Dataset Condensation with Latent Space Knowledge Factorization and Sharing](https://arxiv.org/abs/2208.10494) (Hae Beom Lee & Dong Bok Lee et al., 2022) [:book:](./citations/lee2022kfs.txt)
+ [Slimmable Dataset Condensation](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Slimmable_Dataset_Condensation_CVPR_2023_paper.html) (Songhua Liu et al., CVPR 2023) [:book:](./citations/liu2023slimmable.txt)
+ [Few-Shot Dataset Distillation via Translative Pre-Training](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Few-Shot_Dataset_Distillation_via_Translative_Pre-Training_ICCV_2023_paper.html) (Songhua Liu et al., ICCV 2023) [:book:](./citations/liu2023fewshot.txt)

<a name="generative" />

### Generative Prior
+ [Generalizing Dataset Distillation via Deep Generative Prior](https://arxiv.org/abs/2305.01649) (George Cazenavette et al., CVPR 2023) [:globe_with_meridians:](https://georgecazenavette.github.io/glad/) [:octocat:](https://github.com/georgecazenavette/glad) [:book:](./citations/cazenavette2023glad.txt)
+ [DiM: Distilling Dataset into Generative Model](https://arxiv.org/abs/2303.04707) (Kai Wang & Jianyang Gu et al., 2023) [:octocat:](https://github.com/vimar-gu/DiM) [:book:](./citations/wang2023dim.txt)

<a name="label" />

### Label Distillation
+ [Flexible Dataset Distillation: Learn Labels Instead of Images](https://arxiv.org/abs/2006.08572) (Ondrej Bohdal et al., NeurIPS 2020 Workshop) [:octocat:](https://github.com/ondrejbohdal/label-distillation) [:book:](./citations/bohdal2020flexible.txt)
+ [Soft-Label Dataset Distillation and Text Dataset Distillation](https://arxiv.org/abs/1910.02551) (Ilia Sucholutsky et al., IJCNN 2021) [:octocat:](https://github.com/ilia10000/dataset-distillation) [:book:](./citations/sucholutsky2021soft.txt)

<a name="quant" />

### Dataset Quantization
+ [Dataset Quantization](https://arxiv.org/abs/2308.10524) (Daquan Zhou & Kai Wang & Jianyang Gu et al., ICCV 2023) [:octocat:](https://github.com/magic-research/Dataset_Quantization) [:book:](./citations/zhou2023dataset.txt)

<a name="multi" />

### Multimodal Distillation
+ [Vision-Language Dataset Distillation](https://arxiv.org/abs/2308.07545) (Xindi Wu et al., 2023) [:globe_with_meridians:](https://princetonvisualai.github.io/multimodal_dataset_distillation/) [:octocat:](https://github.com/princetonvisualai/multimodal_dataset_distillation) [:book:](./citations/wu2023multi.txt)

<a name="benchmark" />

### Benchmark

+ [DC-BENCH: Dataset Condensation Benchmark](https://arxiv.org/abs/2207.09639) (Justin Cui et al., NeurIPS 2022) [:globe_with_meridians:](https://dc-bench.github.io/) [:octocat:](https://github.com/justincui03/dc_benchmark) [:book:](./citations/cui2022dc.txt)
+ [A Comprehensive Study on Dataset Distillation: Performance, Privacy, Robustness and Fairness](https://arxiv.org/abs/2305.03355)) (Zongxiong Chen & Jiahui Geng et al., 2023) [:book:](./citations/chen2023study.txt)

<a name="survey" />

### Survey

+ [Data Distillation: A Survey](https://arxiv.org/abs/2301.04272) (Noveen Sachdeva et al., TMLR 2023) [:book:](./citations/sachdeva2023survey.txt)
+ [A Survey on Dataset Distillation: Approaches, Applications and Future Directions](https://arxiv.org/abs/2305.01975) (Jiahui Geng & Zongxiong Chen et al., IJCAI 2023) [:book:](./citations/geng2023survey.txt)
+ [A Comprehensive Survey to Dataset Distillation](https://arxiv.org/abs/2301.05603) (Shiye Lei et al., TPAMI 2023) [:book:](./citations/lei2023survey.txt)
+ [Dataset Distillation: A Comprehensive Review](https://arxiv.org/abs/2301.07014) (Ruonan Yu & Songhua Liu et al., TPAMI 2023) [:book:](./citations/yu2023review.txt)

## Applications

<a name="continual" />

### Continual Learning
+ [Reducing Catastrophic Forgetting with Learning on Synthetic Data](https://arxiv.org/abs/2004.14046) (Wojciech Masarczyk et al., CVPR 2020 Workshop) [:book:](./citations/masarczyk2020reducing.txt)
+ [Condensed Composite Memory Continual Learning](https://arxiv.org/abs/2102.09890) (Felix Wiewel et al., IJCNN 2021) [:octocat:](https://github.com/FelixWiewel/CCMCL) [:book:](./citations/wiewel2021soft.txt)
+ [Distilled Replay: Overcoming Forgetting through Synthetic Samples](https://arxiv.org/abs/2103.15851) (Andrea Rosasco et al., IJCAI 2021 Workshop) [:octocat:](https://github.com/andrearosasco/DistilledReplay) [:book:](./citations/rosasco2021distilled.txt)
+ [Sample Condensation in Online Continual Learning](https://arxiv.org/abs/2206.11849) (Mattia Sangermano et al., IJCNN 2022) [:book:](./citations/sangermano2022sample.txt)
+ [Summarizing Stream Data for Memory-Restricted Online Continual Learning](https://arxiv.org/abs/2305.16645) (Jianyang Gu et al., 2023) [:octocat:](https://github.com/vimar-gu/SSD) [:book:](./citations/gu2023ssd.txt)

<a name="privacy" />

### Privacy
+ [SecDD: Efficient and Secure Method for Remotely Training Neural Networks](https://arxiv.org/abs/2009.09155) (Ilia Sucholutsky et al., AAAI 2021) [:book:](./citations/sucholutsky2021secdd.txt)
+ [Privacy for Free: How does Dataset Condensation Help Privacy?](https://arxiv.org/abs/2206.00240) (Tian Dong et al., ICML 2022) [:book:](./citations/dong2022privacy.txt)
+ [Can We Achieve Robustness from Data Alone?](https://arxiv.org/abs/2207.11727) (Nikolaos Tsilivis et al., ICML 2022 Workshop) [:book:](./citations/tsilivis2022robust.txt)
+ [Private Set Generation with Discriminative Information](https://arxiv.org/abs/2211.04446) (Dingfan Chen et al., NeurIPS 2022) [:octocat:](https://github.com/DingfanChen/Private-Set) [:book:](./citations/chen2022privacy.txt)
+ [Towards Robust Dataset Learning](https://arxiv.org/abs/2211.10752) (Yihan Wu et al., 2022) [:book:](./citations/wu2022towards.txt)
+ [Backdoor Attacks Against Dataset Distillation](https://arxiv.org/abs/2301.01197) (Yugeng Liu et al., NDSS 2023) [:octocat:](https://github.com/liuyugeng/baadd) [:book:](./citations/liu2023backdoor.txt)
+ [Differentially Private Kernel Inducing Points (DP-KIP) for Privacy-preserving Data Distillation](https://arxiv.org/abs/2301.13389) (Margarita Vinaroz et al., 2023) [:book:](./citations/vinaroz2023dpkip.txt)
+ [Dataset Distillation Fixes Dataset Reconstruction Attacks](https://arxiv.org/abs/2302.01428) (Noel Loo et al., 2023) [:octocat:](https://github.com/yolky/distillation_fixes_reconstruction) [:book:](./citations/loo2023attack.txt)

<a name="medical" />

### Medical
+ [Soft-Label Anonymous Gastric X-ray Image Distillation](https://arxiv.org/abs/2104.02857) (Guang Li et al., ICIP 2020) [:octocat:](https://github.com/Guang000/dataset-distillation) [:book:](./citations/li2020soft.txt) 
+ [Compressed Gastric Image Generation Based on Soft-Label Dataset Distillation for Medical Data Sharing](https://arxiv.org/abs/2209.14635) (Guang Li et al., CMPB 2022) [:octocat:](https://github.com/Guang000/dataset-distillation) [:book:](./citations/li2022compressed.txt)
+ [Dataset Distillation for Medical Dataset Sharing](https://r2hcai.github.io/AAAI-23/pages/accepted-papers.html) (Guang Li et al., AAAI 2023 Workshop) [:octocat:](https://github.com/Guang000/mtt-distillation) [:book:](./citations/li2023sharing.txt)
+ [Communication-Efficient Federated Skin Lesion Classification with Generalizable Dataset Distillation](https://workshop2023.isic-archive.com) (Yuchen Tian & Jiacheng Wang, MICCAI 2023 Workshop) [:book:](./citations/tian2023gdd.txt)

<a name="fed" />

### Federated Learning
+ [Federated Learning via Synthetic Data](https://arxiv.org/abs/2008.04489) (Jack Goetz et al., 2020) [:book:](./citations/goetz2020federated.txt)
+ [Distilled One-Shot Federated Learning](https://arxiv.org/abs/2009.07999) (Yanlin Zhou et al., 2020) [:book:](./citations/zhou2020distilled.txt)
+ [DENSE: Data-Free One-Shot Federated Learning](https://arxiv.org/abs/2112.12371) (Jie Zhang & Chen Chen et al., NeurIPS 2022) [:book:](./citations/zhang2022dense.txt)
+ [FedSynth: Gradient Compression via Synthetic Data in Federated Learning](https://arxiv.org/abs/2204.01273) (Shengyuan Hu et al., 2022) [:book:](./citations/hu2022fedsynth.txt)
+ [DYNAFED: Tackling Client Data Heterogeneity with Global Dynamics](https://arxiv.org/abs/2211.10878) (Renjie Pi et al., 2022) [:book:](./citations/pi2022dynafed.txt)
+ [Meta Knowledge Condensation for Federated Learning](https://arxiv.org/abs/2209.14851) (Ping Liu et al., ICLR 2023) [:book:](./citations/liu2023meta.txt)
+ [FedDM: Iterative Distribution Matching for Communication-Efficient Federated Learning](https://arxiv.org/abs/2207.09653) (Yuanhao Xiong & Ruochen Wang et al., CVPR 2023) [:book:](./citations/xiong2023feddm.txt)
+ [Federated Learning via Decentralized Dataset Distillation in Resource-Constrained Edge Environments](https://arxiv.org/abs/2208.11311) (Rui Song et al., IJCNN 2023) [:book:](./citations/song2023federated.txt)
+ [Fed-GLOSS-DP: Federated, Global Learning using Synthetic Sets with Record Level Differential Privacy](https://arxiv.org/abs/2302.01068) (Hui-Po Wang et al., 2023) [:book:](./citations/wang2023fed.txt)
+ [Federated Virtual Learning on Heterogeneous Data with Local-global Distillation](https://arxiv.org/abs/2303.02278) (Chun-Yin Huang et al., 2023) [:book:](./citations/huang2023federated.txt)

<a name="gnn" />

### Graph Neural Network
+ [Graph Condensation for Graph Neural Networks](https://arxiv.org/abs/2110.07580) (Wei Jin et al., ICLR 2022) [:octocat:](https://github.com/chandlerbang/gcond) [:book:](./citations/jin2022graph.txt)
+ [Condensing Graphs via One-Step Gradient Matching](https://arxiv.org/abs/2206.07746) (Wei Jin et al., KDD 2022) [:octocat:](https://github.com/amazon-research/DosCond) [:book:](./citations/jin2022condensing.txt)
+ [Graph Condensation via Receptive Field Distribution Matching](https://arxiv.org/abs/2206.13697) (Mengyang Liu et al., 2022) [:book:](./citations/liu2022graph.txt)
+ [CaT: Balanced Continual Graph Learning with Graph Condensation](https://arxiv.org/abs/2309.09455) (Liu Yilun et al., ICDM 2023) [:octocat:](https://github.com/superallen13/CaT-CGL) [:book:](./citations/liu2023cat.txt)
+ [Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data](https://arxiv.org/abs/2306.02664) (Xin Zheng et al., NeurIPS 2023) [:book:](./citations/zheng2023sfgc.txt)
+ [Does Graph Distillation See Like Vision Dataset Counterpart?](https://arxiv.org/abs/2310.09192) (Beining Yang & Kai Wang et al., NeurIPS 2023) [:octocat:](https://github.com/RingBDStack/SGDD) [:book:](./citations/yang2023sgdd.txt)

<a name="nas" />

### Neural Architecture Search
+ [Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://arxiv.org/abs/1912.07768) (Felipe Petroski Such et al., ICML 2020) [:octocat:](https://github.com/uber-research/GTN) [:book:](./citations/such2020generative.txt)
+ [Learning to Generate Synthetic Training Data using Gradient Matching and Implicit Differentiation](https://arxiv.org/abs/2203.08559) (Dmitry Medvedev et al., AIST 2021) [:octocat:](https://github.com/dm-medvedev/efficientdistillation) [:book:](./citations/medvedev2021tabular.txt)

<a name="fashion" />

### Fashion, Art, and Design
+ [Wearable ImageNet: Synthesizing Tileable Textures via Dataset Distillation](https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/html/Cazenavette_Wearable_ImageNet_Synthesizing_Tileable_Textures_via_Dataset_Distillation_CVPRW_2022_paper.html) (George Cazenavette et al., CVPR 2022 Workshop) [:globe_with_meridians:](https://georgecazenavette.github.io/mtt-distillation/) [:octocat:](https://github.com/georgecazenavette/mtt-distillation) [:book:](./citations/cazenavette2022textures.txt)
+ [Learning from Designers: Fashion Compatibility Analysis Via Dataset Distillation](https://ieeexplore.ieee.org/document/9897234) (Yulan Chen et al., ICIP 2022) [:book:](./citations/chen2022fashion.txt)

<a name="kd" />

### Knowledge Distillation
+ [Knowledge Condensation Distillation](https://arxiv.org/abs/2207.05409) (Chenxin Li et al., ECCV 2022) [:octocat:](https://github.com/dzy3/KCD) [:book:](./citations/li2022knowledge.txt)

<a name="rec" />

### Recommender Systems
+ [Infinite Recommendation Networks: A Data-Centric Approach](https://arxiv.org/abs/2206.02626) (Noveen Sachdeva et al., NeurIPS 2022) [:octocat:](https://github.com/noveens/distill_cf) [:book:](./citations/sachdeva2022data.txt)
+ [Gradient Matching for Categorical Data Distillation in CTR Prediction](https://dl.acm.org/doi/10.1145/3604915.3608769) (Chen Wang et al., RecSys 2023) [:book:](./citations/wang2023cgm.txt)

<a name="blackbox" />

### Blackbox Optimization
+ [Bidirectional Learning for Offline Infinite-width Model-based Optimization](https://arxiv.org/abs/2209.07507) (Can Chen et al., NeurIPS 2022) [:octocat:](https://github.com/ggchen1997/bdi) [:book:](./citations/chen2022bidirectional.txt) 
+ [Bidirectional Learning for Offline Model-based Biological Sequence Design](https://arxiv.org/abs/2301.02931) (Can Chen et al., ICML 2023) [:octocat:](https://github.com/GGchen1997/BIB-ICML2023-Submission) [:book:](./citations/chen2023bidirectional.txt)

<a name="trustworthy" />

### Trustworthy
+ [Rethinking Data Distillation: Do Not Overlook Calibration](https://arxiv.org/abs/2307.12463) (Dongyao Zhu et al., ICCV 2023) [:book:](./citations/zhu2023calibration.txt)
+ [Towards Trustworthy Dataset Distillation](https://arxiv.org/abs/2307.09165) (Shijie Ma et al., 2023) [:book:](./citations/ma2023trustworthy.txt)

<a name="retrieval" />

### Retrieval
+ [Towards Efficient Deep Hashing Retrieval: Condensing Your Data via Feature-Embedding Matching](https://arxiv.org/abs/2305.18076) (Tao Feng & Jie Zhang et al., 2023) [:book:](./citations/feng2023hash.txt)

<a name="text" />

### Text
+ [Data Distillation for Text Classification](https://arxiv.org/abs/2104.08448) (Yongqi Li et al., 2021) [:book:](./citations/li2021text.txt)
+ [Dataset Distillation with Attention Labels for Fine-tuning BERT](https://aclanthology.org/2023.acl-short.12/) (Aru Maekawa et al., ACL 2023) [:octocat:](https://github.com/arumaekawa/dataset-distillation-with-attention-labels) [:book:](./citations/maekawa2023text.txt)

<a name="tabular" />

### Tabular
+ [New Properties of the Data Distillation Method When Working With Tabular Data](https://arxiv.org/abs/2010.09839) (Dmitry Medvedev et al., AIST 2020) [:octocat:](https://github.com/dm-medvedev/dataset-distillation) [:book:](./citations/medvedev2020tabular.txt)

<a name="media" />

## Media Coverage
+ [Beginning of Awesome-Dataset-Distillation](https://twitter.com/TongzhouWang/status/1560043815204970497?cxt=HHwWgoCz9bPlsaYrAAAA)
+ [Most Popular AI Research Aug 2022](https://www.libhunt.com/posts/874974-d-most-popular-ai-research-aug-2022-ranked-based-on-github-stars)
+ [一个项目帮你了解数据集蒸馏Dataset Distillation](https://www.jiqizhixin.com/articles/2022-10-11-22)
+ [浓缩就是精华：用大一统视角看待数据集蒸馏](https://mp.weixin.qq.com/s/__IjS0_FMpu35X9cNhNhPg)

<a name="ack" />

## Acknowledgments
We want to thank [Nikolaos Tsilivis](https://github.com/Tsili42), [Wei Jin](https://github.com/ChandlerBang), [Yongchao Zhou](https://github.com/yongchao97), [Noveen Sachdeva](https://github.com/noveens), [Can Chen](https://github.com/GGchen1997), [Guangxiang Zhao](https://github.com/zhaoguangxiang), [Shiye Lei](https://github.com/LeavesLei), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/), [Dmitry Medvedev](https://github.com/dm-medvedev), [Seungjae Shin](https://github.com/SJShin-AI), [Jiawei Du](https://github.com/AngusDujw), [Yidi Jiang](https://github.com/Jiang-Yidi), [Xindi Wu](https://github.com/XindiWu), [Guangyi Liu](https://github.com/lgy0404), [Yilun Liu](https://github.com/superallen13), and [Kai Wang](https://github.com/kaiwang960112) for their valuable suggestions and contributions.
