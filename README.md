[IN PROGRESS] Code + Models for EMNLP 2022 Findings paper "Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure"
Paper Link: https://arxiv.org/abs/2211.07743

Model Checkpoints (Highest-performing model amongst the 5 random seeds)

```
GEN_SCL_NAT-RESTAURANT
GEN_SCL_NAT-LAPTOP
GEN_SCL_NAT-LAPTOP-L1
```

Module Requirements:

```
Python 3.9+
Pytorch 1.10
Ptorch Lightning 1.8
```

See `requirements.txt` for reproducing full environment
You can initialize an environment by doing something like 
```
conda create environment.yml
conda activate gen_scl_nat_env
```

Module Usage.
1. Initialize + activate conda environment
2. Download data + models to `data/` and `/trained_models` folders
    TODO: Add wget-able links instead of GDrive
3. Run `main_gen_scl_nat.py` with specified json configuration file (see `configs/`).   
e.g. `python main_gen_scl_nat.py --config <config_file>`

Config file examples:

Train + Evaluate model:
```
{
    `dataset_path` # path to folder containing dataset (split into train/validation/test).  
    `model_path` # path to t5 (training) or fine-tuned model (evaluation) 
    `train`: { 
        `num_epochs: 30` 
        `output_path: <output_path>` 
        . 
        . 
        `<training parameters>` 
    } 
    `evaluate`: { 
        `num_beams`: 5 
        `print_metrics`: True 
        `evaluation_log_path`:  # optional 
    } 
}
```
 

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
