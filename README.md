Code + Models for EMNLP 2022 Findings paper "Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure"
Paper Link: https://arxiv.org/abs/2211.07743

![image](https://user-images.githubusercontent.com/22947125/224211557-5f9b37ac-10ce-4e1b-8c53-a574233d94d1.png)


Pre-trained Model Usage:

The following trained models are available for download on Google Drive (highest-performing model amongst the 5 random seeds):
```
GEN_SCL_NAT-RESTAURANT
GEN_SCL_NAT-LAPTOP
GEN_SCL_NAT-LAPTOP-L1
```

Drive link: https://drive.google.com/drive/folders/1g30oS8hpqn6tAGNyLbOwEoLLmhHOy94o?usp=share_link


Module Requirements:

You can recreate the full Conda environment used by running the following (may require some tweaking of the environment name/path to run on your machine):
```
conda env create -f environment.yml
conda activate gen_scl_nat_env
```

Otherwise, key dependencies used are listed here:

```
Python >= 3.9+
torch >= 1.10
pytorch-lightning >= 1.8.6
sentencepiece >= 0.1.97
transformers >= 4.19.0
```

Module Usage:
1. Initialize + activate conda environment
2. Download and untar trained models to `models/`
3. Run `main_gen_scl_nat.py` for model training/inference. 
	`configs/` contains example scripts for running evaluation on each model from the paper

Please cite our paper as such:
```
@InProceedings{peper22generativeacos,
  author = 	"Peper, Joseph J.
			and Wang, Lu",
  title = 	"Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure",
  booktitle = 	"Conference on Empirical Methods in Natural Language Processing",
  year = 	"2022"
}
```
