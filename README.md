## REANO: Optimizing Retrieval-Augmented Reader Models through Knowledge Graph Generation 

This repository contains the PyTorch implementation for our REANO

Further details about REANO can be found in our paper.

## Requirements
* python 3.8
* PyTorch 1.13.1 
* transformers 4.31.0 
* scipy 1.10.1 
* Spacy 3.4.0

## File Structure:
```bash
project-root/
├── common/               # to store datasets and models
├── src/                  # Data processing, evaluation and other untils
├───── main.py            # train and evaluate the model.
```

## Run the demo

<!-- To perform semi-supervised object classification on Cora dataset, run the following command: -->
We provide the pre-processed data of the 2wikimultihopqa dataset at [here](https://drive.google.com/drive/folders/1JkinZgtO5SCip_E4KFm4VjC7oXLSFR4N?usp=sharing).

To train and evaluate the performance of REANO on the 2wikimultihopqa dataset, run the following command: 

```bash
python main.py \
    --train_data /path/to/train/data \
    --eval_data /path/to/dev/data \ 
    --model_size base \
    --text_maxlength 250 \
    --k 20 \
    --hop 3 \
    --n_context 10 \
    --name wikimultihopqa_base_version \
    --checkpoint_dir checkpoint \
    --lr 1e-4 \ 
    --scheduler fixed \
    --weight_decay 0.01 \
    --per_gpu_batch_size 8 \
    --accumulation_steps 4 \
    --clip 1.0 \
    --total_steps 15000 \
    --eval_freq 500 \
    --save_freq 15000 
```
