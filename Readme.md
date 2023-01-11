# HADA: A Graph-based Amalgamation Framework in Image-Text Retrieval
This is a sub-repository of the [HADA](https://github.com/m2man/HADA) where we applied HADA on-the-top of [LAVIS library](https://github.com/salesforce/LAVIS).

## Introduction
Our paper has been accepted at ECIR'23

HADA is a framework that combines any pretrained SOTA models in image-text retrieval (ITR) to produce a better result. A special feature in HADA is that this framework only introduces a tiny number of additonal trainable parameters. Thus, it does not required multiple GPUs to train HADA or external large-scale dataset (although pretraining may further improve the performance).

In this sub-repo, we used HADA to combine 2 SOTA models including BLIP and CLIP from [LAVIS library](https://github.com/salesforce/LAVIS). The total recall was increase by 3.94% on the Flickr30k dataset and 1.49% on MSCOCO. 

## Installation
Please install [LAVIS](https://github.com/salesforce/LAVIS) before running.

We used **mlflow-ui** to keep track the performance between configurations. Please modify or remove this related-part if you do not want to use.