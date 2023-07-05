# Men Also Do Laundry: Multi-Attribute Bias Amplification
Repository for accepted ICML 2023 paper ["Men Also Do Laundry: Multi-Attribute Bias Amplification"](https://arxiv.org/abs/2210.11924), which presents interpretable metrics for measuring bias amplification from multiple attributes

[[Project Page]](https://sonyresearch.github.io/multi_bias_amp/) &nbsp; | &nbsp;  [[arXiv]](https://arxiv.org/abs/2210.11924)
## Abstract 
> As computer vision systems become more widely deployed, there is increasing concern from both the research community and the public that these systems are not only reproducing but amplifying harmful social biases. The phenomenon of bias amplification, which is the focus of this work, refers to models amplifying inherent training set biases at test time. Existing metrics measure bias amplification with respect to single annotated attributes (e.g., 𝚌𝚘𝚖𝚙𝚞𝚝𝚎𝚛). However, several visual datasets consist of images with multiple attribute annotations. We show models can learn to exploit correlations with respect to multiple attributes (e.g., {𝚌𝚘𝚖𝚙𝚞𝚝𝚎𝚛, 𝚔𝚎𝚢𝚋𝚘𝚊𝚛𝚍}), which are not accounted for by current metrics. In addition, we show current metrics can give the erroneous impression that minimal or no bias amplification has occurred as they involve aggregating over positive and negative values. Further, these metrics lack a clear desired value, making them difficult to interpret. To address these shortcomings, we propose a new metric: Multi-Attribute Bias Amplification. We validate our proposed metric through an analysis of gender bias amplification on the COCO and imSitu datasets. Finally, we benchmark bias mitigation methods using our proposed metric, suggesting possible avenues for future bias mitigation
---

## Setup

To install the necessary packages, use the following command:

```
    pip install -r requirements.txt 
```

Requirements include Python >=3.6, numpy, scikit-learn, and tqdm. 

## Metrics

All of the necessary code files are in ``metrics/``. There are three metrics we provide the implementations for as follows:

- ``mals.py``: Undirected multi-attribute bias amplification from [Zhao et al.](https://aclanthology.org/D17-1323/)
- ``dba.py``: Directed multi-attribute bias amplification from [Wang and Russakovsky](https://proceedings.mlr.press/v139/wang21t.html)
- ``mba.py``: Undirected and directed multi-attribute bias amplficiation

## Bibtex 

```
@inproceedings{zhao2023men,
    title={Men Also Do Laundry: Multi-Attribute Bias Amplification},
    author={Zhao, Dora and Andrews, Jerone TA and Xiang, Alice},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2023}
}
```
---
## Contact 

For questions, please contact Dora Zhao (dora.zhao@sony.com)