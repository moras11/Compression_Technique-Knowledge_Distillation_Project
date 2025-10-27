# Compression Techniques – Knowledge Distillation Project

This repository contains our group project on **Knowledge Distillation**, implemented as part of the *Scalable Computing for Data Analytics* module at **University College Cork (UCC)**.  
The objective was to explore model compression techniques that transfer knowledge from a large (teacher) network to a smaller (student) network while maintaining comparable accuracy and faster inference.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Training Pipeline](#training-pipeline)
- [Results](#results)
- [Setup & Installation](#setup--installation)
- [Pretrained Models](#pretrained-models)
- [Team Members](#team-members)
- [References](#references)
- [Acknowledgment](#acknowledgment)

---

## Overview
Knowledge Distillation (KD) is a model compression approach where a smaller *student* model learns from a larger, pretrained *teacher* model by mimicking its soft output probabilities or intermediate representations.

In this project, we:
- Implemented **Soft-Label Distillation**, **Attention Transfer**, and **Deep Mutual Learning**.
- Trained and evaluated models on **CIFAR-10** and **CelebA** datasets.
- Compared model size, training time, and inference accuracy across techniques.

---

## Architecture
We experimented with the following configurations:

| Component | Description |
|------------|-------------|
| **Teacher Model** | ResNet-50 |
| **Student Model** | ResNet-18 |
| **Distillation Methods** | Soft Targets (Hinton et al.), Attention Transfer, Deep Mutual Learning |

```text
Teacher (ResNet50)
        ↓
    Soft logits
        ↓
Student (ResNet18)

Datasets
Dataset	Description	Classes	Images
CIFAR-10	32×32 color images	10	60,000
CelebA	Celebrity attributes dataset	40 attributes	200,000+

Dataset structure:

kotlin
Copy code
Cifar10/
 ├── data/
 ├── models/
 └── results/
celebA/
 ├── data/
 ├── models/
 └── results/
Training Pipeline
Train Teacher Model on the dataset.

Use soft labels or attention maps from the teacher to train the Student Model.

Evaluate accuracy, loss, and inference time.

Compare model compression ratio and performance.

Key scripts

app.py → main execution script

requirements.txt → dependencies

Cifar10/ & celebA/ → dataset-specific training & model code

Results Summary
Method	Dataset	Teacher Acc	Student Acc	Compression	Notes
Soft Distillation	CIFAR-10	91.8%	88.5%	~3.2×	Smooth transfer
Attention Transfer	CelebA	92.0%	89.7%	~3.0×	Strong feature alignment
Deep Mutual Learning	CIFAR-10	91.8%	89.3%	~3.1×	Cooperative training

Setup & Installation
Requirements
Python 3.8+

PyTorch ≥ 1.10

torchvision

numpy, matplotlib, tqdm

Installation
bash
Copy code
# Clone this repository
git clone https://github.com/moras11/Compression_Technique-Knowledge_Distillation_Project.git
cd Compression_Technique-Knowledge_Distillation_Project

# Install dependencies
pip install -r requirements.txt

# Run example training
python app.py --dataset cifar10 --mode distill
Pretrained Models
Pretrained weights are not included in this repository due to GitHub size limits.
You can download them from the link below and place them in their respective folders:

📚 References
Hinton, G. et al. Distilling the Knowledge in a Neural Network, 2015.

Zagoruyko, S., Komodakis, N. Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer, 2016.

Zhang, Y. et al. Deep Mutual Learning, 2018.