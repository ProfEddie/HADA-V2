# HADA: A Graph-based Amalgamation Framework in Image-Text Retrieval
This is a sub-repository of the [HADA](https://github.com/m2man/HADA) where we applied HADA on-the-top of [LAVIS library](https://github.com/salesforce/LAVIS).

## Introduction
Our [paper](https://link.springer.com/chapter/10.1007/978-3-031-28244-7_45) has been accepted at ECIR'23

HADA is a framework that combines any pretrained SOTA models in image-text retrieval (ITR) to produce a better result. A special feature in HADA is that this framework only introduces a tiny number of additonal trainable parameters. Thus, it does not required multiple GPUs to train HADA or external large-scale dataset (although pretraining may further improve the performance).

In this sub-repo, we used HADA to combine 2 SOTA models including BLIP and CLIP from [LAVIS library](https://github.com/salesforce/LAVIS). The total recall was increase by 3.94% on the Flickr30k dataset and 1.49% on MSCOCO. 

## Installation
Please install [LAVIS](https://github.com/salesforce/LAVIS) before running.

We used **mlflow-ui** to keep track the performance between configurations. Please modify or remove this related-part if you do not want to use.

## Pretrained Models
We uploaded the pretrained models [here](https://drive.google.com/drive/folders/17_AYT9wNiVNAgZdgClMRDewcmzZye66d?usp=share_link).

## Train and Evaluate
Remember to update the path in the config files in **Config** folders. Then you can train or evaluate by the file `run_exp.py`

```python
# Train (2 steps, 1st step for initial learning and 2nd step for finetuning)
# Flickr
python main.py -cp Config/flickr30k/C1.yml -rm train
python main.py -cp Config/flickr30k/C1_cont.yml -rm train

# MSCOCO
python main.py -cp Config/mscoco/C1.yml -rm train
python main.py -cp Config/mscoco/C1_cont_Argment.yml -rm train


# Test (skip train if you download pretrained model)
# Flickr
python main.py -cp Config/flickr30k/C1_cont.yml -rm test

# MSCOCO
python main.py -cp Config/mscoco/C1_cont_Argment.yml -rm test
```

## Contact
For any issue or comment, you can directly email me at manh.nguyen5@mail.dcu.ie