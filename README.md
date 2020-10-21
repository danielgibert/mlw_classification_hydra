 # HYDRA: A multimodal deep learning framework for malware classification

This code base is no longer maintained and exists as a historical artifact to supplement
the paper [HYDRA: A multimodal deep learning framework for malware classification](https://www.sciencedirect.com/science/article/pii/S0167404820301462).


## Requirements

Code is written in Python 3.6 and requires Tensorflow==2.3.0

## Citing 
If you find this work useful in your research, please consider citing:
```
@article{GIBERT2020101873,
title = "HYDRA: A multimodal deep learning framework for malware classification",
journal = "Computers & Security",
volume = "95",
pages = "101873",
year = "2020",
issn = "0167-4048",
doi = "https://doi.org/10.1016/j.cose.2020.101873",
url = "http://www.sciencedirect.com/science/article/pii/S0167404820301462",
author = "Daniel Gibert and Carles Mateu and Jordi Planes",
keywords = "Malware classification, Machine learning, Deep learning, Feature fusion, Multimodal learning",
abstract = "While traditional machine learning methods for malware detection largely depend on hand-designed features, which are based on experts’ knowledge of the domain, end-to-end learning approaches take the raw executable as input, and try to learn a set of descriptive features from it. Although the latter might behave badly in problems where there are not many data available or where the dataset is imbalanced. In this paper we present HYDRA, a novel framework to address the task of malware detection and classification by combining various types of features to discover the relationships between distinct modalities. Our approach learns from various sources to maximize the benefits of multiple feature types to reflect the characteristics of malware executables. We propose a baseline system that consists of both hand-engineered and end-to-end components to combine the benefits of feature engineering and deep learning so that malware characteristics are effectively represented. An extensive analysis of state-of-the-art methods on the Microsoft Malware Classification Challenge benchmark shows that the proposed solution achieves comparable results to gradient boosting methods in the literature and higher yield in comparison with deep learning approaches."
}
```

## ToDo
* Transfer the weights of the individually trained subcomponents
* Modality dropout.