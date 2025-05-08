## VLEM
[CogSci 2025] Official pytorch implementation of ["Towards a Vision-Language Episodic Memory: Large-scale Pretrained Model-Augmented Hippocampal Attractor Dynamics"]()

### Abstract

<img src=".\assets\method.jpeg" alt="method" style="zoom: 15%;" />

Modeling episodic memory (EM) remains a significant challenge in both neuroscience and AI, with existing models either lacking interpretability or struggling with practical applications. This paper proposes the Vision-Language Episodic Memory (***VLEM***) framework to address these challenges by integrating large-scale pretrained models with hippocampal attractor dynamics. VLEM leverages the strong semantic understanding of pretrained models to transform sensory input into semantic embeddings as the neocortex, while the hippocampus supports stable memory storage and retrieval through attractor dynamics. In addition, VLEM incorporates prefrontal working memory and the entorhinal gateway, allowing interaction between the neocortex and the hippocampus. To facilitate real-world applications, we introduce EpiGibson, a 3D simulation platform for generating episodic memory data. Experimental results demonstrate the VLEM framework's ability to efficiently learn high-level temporal representations from sensory input, showcasing its robustness, interpretability, and applicability in real-world scenarios.

### Installation

Download this repository.

```
git clone https://github.com/fudan-birlab/VLEM.git
cd VLEM
```

#### EpiGibson

Create the environment for EpiGibson.

```
conda create -n omnigibson python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
conda activate omnigibson
cd EpiGibson/OmniGibson
pip install -e .
python -m omnigibson.install
cp -r ../og_dataset ./omnigibson/data/
```

#### VLEM

Create the environment for VLEM.

```
conda create -n VLEM python=3.10
conda activate VLEM
pip install -r requirements.txt
```

### Getting Start

#### EpiGibson Dataset Generation

An NVIDIA **RTX** GPU is required (e.g., RTX A4500).

```
cd EpiGibson
python -u ./generate_dataset.py
```

#### VLEM

The dataset type can be configured as either `EpiGibson` or `patternDataset` (with adjustable data scale for `patternDataset`). The model type can be set to either `distribute` or `merge`, where `distribute` corresponds to the `VLEM` model described in the paper, and `merge` corresponds to the `VLEM (merge)` variant.

##### 1. Training

```
python -u main.py
```

##### 2. Evaluation

```
python -u test.py
```

##### 3. Visualization

See `evaluation.ipynb` & `visualization.ipynb`.

### Results

<img src=".\assets\eval-results.jpeg" alt="eval-results" style="zoom:30%;" />

<img src=".\assets\results.jpeg" alt="within-Wen" style="zoom:30%;" />

<!-- ### Citation

If you find our work useful to your research, please consider citing:

```
``` -->

### Acknowledgement

EpiGibson is built on [OmniGibson](https://github.com/StanfordVL/OmniGibson). Thanks for their excellent work.
