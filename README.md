# [ReMoDiff - Restoring Missing Modalities with Diffusion](https://openreview.net/pdf?id=BuGFwUS9B3)

![](https://img.shields.io/badge/Platform-PyTorch-blue)
![](https://img.shields.io/badge/Language-Python-{green}.svg)

We propose the Restoring Missing Modalities with Diffusion (ReMoDiff) method that maps input random noise to the distribution space of missing modalities and recovers missing data in accordance with their original distributions. 

## The Framework of ReMoDiff.

![](ReMoDiff.png)

(1) ReMoDiff maps input random noise to the distribution space of missing modalities and recovers missing data in accordance with their original distributions.

(2) To minimize the semantic ambiguity between the missing and recovered modalities, ReMoDiff utilize the available modalities as prior conditions to guide and refine the recovering process.

## Usage

### Datasets
We use the ChaLearn First Impressions V2 dataset.

### Run the Codes
Running the following command:
```
python train.py
```
