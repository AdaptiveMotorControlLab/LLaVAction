# LLaVAction: Evaluating and Training Multi-Modal Large Language Models for Action Recognition


- This repository contains the implementation for our ICCV 2025 submission on evaluating and training multi-modal large language models for action recognition. 
- Our code is built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), and files in the directory `llavaction/action` are related to this work. We thank the authors of LLaVA-NeXT for making their code publicly available.
- The files in the `/eval`, `/model`, `/serve` and `/train` are directly from [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT), unless modified and noted below.
- Modified files are:
  - - /model/llava_arch.py
  - - /model/language_model/llava_qwen.py
  - - /train/train.py
  - - /train/llava_trainer.py
  - - /utils.py
  - - A diff can be generated against the commit (79ef45a6d8b89b92d7a8525f077c3a3a9894a87d) of LLaVA-NeXT to see our modifications.
- The code will be made publicly available when published. For review, the provided code and model license is [no license](https://choosealicense.com/no-permission/).


## Demo 
- Currently, we provide code to run video inference in a Jupyter Notebook (which can be run on Google Colaboratory).
**Installation guide for video inference:**
```bash
conda create -n llavaction python=3.10 -y
conda activate llavaction
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e .
```

- Please see the `/example` directory for a demo notebook.
