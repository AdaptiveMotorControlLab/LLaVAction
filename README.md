# LLaVAction: Evaluating and Training Multi-Modal Large Language Models for Action Recognition

[![Static Badge](https://img.shields.io/badge/LLaVAction-paper-green)](http://arxiv.org/abs/tbd)
[![Demo Website](https://img.shields.io/badge/LLaVAction-website-red)](https://mmathislab.github.io/llavaction/)
[![llavaction-checkpoints](https://img.shields.io/badge/LLaVAction-checkpoints_🤗-blue)](https://huggingface.co/MLAdaptiveIntelligence)

[![Downloads](https://static.pepy.tech/badge/llavaction)](https://pepy.tech/project/llavaction)
[![Downloads](https://static.pepy.tech/badge/llavaction/month)](https://pepy.tech/project/llavaction)
[![PyPI version](https://badge.fury.io/py/llavaction.svg)](https://badge.fury.io/py/llavaction)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-red)

## Abstract

Understanding human behavior requires measuring behavioral actions. Due to its complexity, behavior is best mapped onto a rich, semantic structure such as language. The recent development of multi-modal large language models (MLLMs) is a promising candidate for a wide range of action understanding tasks. In this work, we focus on evaluating and then improving MLLMs to perform action recognition. We reformulate EPIC-KITCHENS-100, one of the largest and most challenging egocentric action datasets, to the form of video multiple question answering (EPIC-KITCHENS-100-MQA). We show that when we sample difficult incorrect answers as distractors, leading MLLMs struggle to recognize the correct actions. We propose a series of methods that greatly improve the MLLMs' ability to perform action recognition, achieving state-of-the-art on both the EPIC-KITCHENS-100 Challenge, as well as outperforming GPT-4o by 21 points in accuracy on EPIC-KITCHENS-100-MQA. Lastly, we show improvements on other action-related video benchmarks such as VideoMME, PerceptionTest and MVBench.

## Code

- This repository contains the implementation for our preprint on evaluating and training multi-modal large language models for action recognition. 
- Our code is built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), and files in the directory `llavaction/action` are related to our work. We thank the authors of LLaVA-NeXT for making their code publicly available.
- The files in the `/eval`, `/model`, `/serve` and `/train` are directly from [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), unless modified and noted below.
  - `/model/llava_arch.py`  
  - `/model/language_model/llava_qwen.py`  
  - `/train/train.py`  
  - `/train/llava_trainer.py`  
  - `/utils.py` 

## Demo 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AdaptiveMotorControlLab/LLaVAction/blob/main/example/llavaction_video_demo.ipynb)
 We provide code to run video inference in a Jupyter Notebook (which can be run on Google Colaboratory).

  
### Installation guide for video inference:
```bash
conda create -n llavaction python=3.10 -y
conda activate llavaction
pip install --upgrade pip  # Enable PEP 660 support.
pip install --pre llavaction
```

- Please see the `/example` directory for a demo notebook.

## EPIC-KITCHENS-100-MQA 

In our work, we introduce a new way to evaluate MLMMs for action recognition by casting EPIC-KITCHENS-100 into a multi-question-answer benchmark. This has not yet been released [as of 3/2025], but please check the issues or open an issue if you are interested in accessing this resource before the paper is published. We also plan to integrate this the package [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

# Acknowledgments 
We thank the Swiss AI Initiative Project ID a03 from the Swiss National Supercomputing Centre (CSCS); Boehringer Ingelheim Fonds PhD stipend (H.Q.); M.W.M. thanks the Vallee Foundation; M.W.M. and A.M. thank the SNSF by grant No. 320030-227871.

![group-logo](https://github.com/user-attachments/assets/ad034dc3-5e92-4e8b-915b-85e443b3bdb2)

