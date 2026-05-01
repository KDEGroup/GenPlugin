# GENPLUGIN
This repository provides a PyTorch reference implementation of the main models and training procedures described in our paper:
> Kun Yang, Siyao Zheng, Tianyi Li, Xiaodong Li, Hui Li.  **GenPlugin: A Plug-and-Play Framework for Long-Tail Generative Recommendation with Exposure Bias Mitigation**.


Within the code directory: the folders ending with "-RAR" use to the Retrieval-Augmented Fine-tuning section, while the rest are for the model pre-training.

You'll need to train a model first, then cache user embeddings before proceeding with the retrieval-augmented fine-tuning.

## Requirements
transformers==4.46.0 

torch==2.3.1+cu121

## Pretrain and Cache



pretrain a model 

``` run_train.sh ```




## RAR finetune 

retrival content aware users

``` python sparse.py ```

use sasrec train a cf model to retrival Collaborative users

run

```run_rag.sh ```

cache user representations and rerank

cd -RAR

finetune model



```run_train.sh ```

test

```run_test.sh```

## Acknowledgements

We greatly appreciate the official [LETTER](https://github.com/HonghuiBao2000/LETTER) and [MQL4Rec](https://github.com/zhaijianyang/MQL4GRec) repository. Our code is built on their framework.
