# -*- coding: utf-8 -*-
"""
Implementation of Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure, an EMNLP 2022 paper.
If using this codebase / models in your work, please cite our work as follows:

@InProceedings{peper22generativeacos,
  author = 	"Peper, Joseph J.
			and Wang, Lu",
  title = 	"Generative Aspect-Based Sentiment Analysis with Contrastive Learning and Expressive Structure",
  booktitle = 	"Conference on Empirical Methods in Natural Language Processing",
  year = 	"2022"
}

Code is largely derived from the codebase used for the EMNLP 2021 paper "Aspect Sentiment Quad Prediction as Paraphrase Generation"
https://github.com/IsakZhang/ABSA-QUAD
"""
import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm
import json
import numpy as np

from torch import nn
from torch.nn.functional import normalize
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from losses import SupConLoss

from transformers import AdamW, T5ForConditionalGeneration, T5Tokenizer
from transformers import get_linear_schedule_with_warmup

from data_utils import GenSCLNatDataset
from data_utils import read_line_examples_from_file
from eval_utils import compute_scores

logger = logging.getLogger(__name__)

DEVICE = f'cuda:{0}'

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--task", default='asqp', type=str, required=True,
                        help="The name of the task, selected from: [asqp, tasd, aste]")
    parser.add_argument("--dataset", default='rest15', type=str, required=True,
                        help="The name of the dataset, selected from: [rest15, rest16]")
    parser.add_argument("--model_name_or_path", default='t5-small', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_direct_eval", action='store_true', 
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--do_inference", action='store_true', 
                        help="Whether to run inference with trained checkpoints")
    # other parameters
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--n_gpu", default=0)
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--num_train_epochs", default=30, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # training details
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--model_prefix", type=str, required=True)
    parser.add_argument("--num_beams", type=int, required=True)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--cont_loss", type=float, default=0.0)
    parser.add_argument("--cont_temp", type=float, default=0.1)
    parser.add_argument('--truncate', action='store_true')


    args = parser.parse_args()

    # create output folder if needed
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    params = [['beams', str(args.num_beams)],
              ['wd', str(args.weight_decay)],
              ['max_epochs', str(args.num_train_epochs)],
              ['es', str(args.early_stopping)],
              ['acc', str(args.gradient_accumulation_steps)],
              ['lr', str(args.learning_rate)],
              ['cont_loss', str(args.cont_loss)],
              ['cont_temp', str(args.cont_temp)],
              ['trunc', str(args.truncate)], # whether to truncate the category labels
              ['seed', str(args.seed)]]

    # TODO CLEANUP TRAINING OUTPUT FOLDER
    # the model path is the prefix
    if args.do_inference and not args.do_train:
        output_fold = args.model_prefix
        print(output_fold)
    else:
        # dump params as part of folder_path
        params = "I".join([elt for elts in params for elt in elts])
        output_fold = "I".join([args.dataset, args.task,args.model_name_or_path, params, args.model_prefix])
        #output_fold = "_".join([args.dataset, args.task, args.model_prefix, args.model_name_or_path])
        print(output_fold)
    output_dir = f"{args.output_folder}/{output_fold}"
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args

def get_dataset(tokenizer, type_path, args):
    return GenSCLNatDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length, task=args.task, truncate=args.truncate)

"""
Uncomment for tsne logging
tsne_dict = {
             'sentiment_vecs': [],
             'opinion_vecs': [],
             'aspect_vecs': [],
             'sentiment_labels': [],
             'opinion_labels': [],
             'aspect_labels': []
             }
"""

class LinearModel(nn.Module):
    """
    Linear models used for the aspect/opinion/sentiment-specific representations
    """
    def __init__(self, model_path):
        super().__init__()
        if model_path == 't5-small':
            self.layer_1 = nn.Linear(512, 1024)
        elif model_path == 't5-base':
            self.layer_1 = nn.Linear(768, 1024)
        else:
            self.layer_1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        """
        Returns an encoding of input X and a simple dropout-perturbed version of X
        For use in the SupConLoss calculation
        """
        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features_summed = torch.sum(last_state, dim=1)
        dropped = self.dropout(features_summed)
        return torch.stack((self.layer_1(features_summed), self.layer_1(dropped)), 1)

class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer, cont_model, op_model, as_model, cat_model):
        super(T5FineTuner, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = tfm_model
        self.cont_model = cont_model
        self.op_model = op_model
        self.as_model = as_model
        self.cat_model = cat_model
        self.tokenizer = tokenizer

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None):
        main_pred = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
        )
       
        last_state = main_pred.encoder_last_hidden_state

        # sentiment contrastive loss
        cont_pred = self.cont_model(last_state, attention_mask)
        # opinion contrastive loss
        op_pred = self.op_model(last_state, attention_mask)
        # aspect contrastive loss
        as_pred = self.as_model(last_state, attention_mask)
        
        # get final encoder layer representation
        masked_last_state = torch.mul(last_state, attention_mask.unsqueeze(-1))
        pooled_encoder_layer = torch.sum(masked_last_state, dim=1)
        pooled_encoder_layer = normalize(pooled_encoder_layer, p=2.0, dim=1)

        return main_pred, cont_pred, op_pred, as_pred, pooled_encoder_layer
        

    def _step(self, batch):
        lm_labels = torch.clone(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs, cont_pred, op_pred, as_pred, pooled_encoder_layer = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
        )

        # define loss with a temperature `temp`
        criterion = SupConLoss(loss_scaling_factor=self.hparams.cont_loss, temperature=self.hparams.cont_temp)
        sentiment_labels = batch['sentiment_labels']
        aspect_labels = batch['aspect_labels']
        opinion_labels = batch['opinion_labels']

        # Calculate the characteristic-specific losses
        cont_summed = cont_pred
        cont_normed = normalize(cont_summed, p=2.0, dim=2)  
        sentiment_contrastive_loss = criterion(cont_normed, sentiment_labels)
        #print('contr_loss:\t', sentiment_contrastive_loss)

        as_summed = as_pred
        as_normed = normalize(as_summed, p=2.0, dim=2)
        aspect_contrastive_loss = criterion(as_normed, aspect_labels)
        #print('as_loss:\t', aspect_contrastive_loss)

        op_summed = op_pred
        op_normed = normalize(op_summed, p=2.0, dim=2)
        opinion_contrastive_loss = criterion(op_normed, opinion_labels)
        #print('op_loss:\t', opinion_contrastive_loss)
        
        """
        Uncomment this section to extract the tsne encodings/labels used for Figure 2 in paper

        # Use these for generating the 'w/ SCL' figures
        sentiment_encs = cont_normed.detach().numpy()[:,0].tolist()
        aspect_encs = as_normed.detach().numpy()[:,0].tolist()
        opinion_encs = op_normed.detach().numpy()[:,0].tolist()
        sentiment_labs = sentiment_labels.detach().tolist()
        aspect_labs = aspect_labels.detach().tolist()
        opinion_labs = opinion_labels.detach().tolist()

        # Use these for the version without SCL (no characteristic-specific representations)
        
        sentiment_encs = pooled_encoder_layer.detach().numpy().tolist()
        aspect_encs = pooled_encoder_layer.detach().numpy().tolist()
        opinion_encs = pooled_encoder_layer.detach().numpy().tolist()
        sentiment_labs = sentiment_labels.detach().tolist()
        aspect_labs = aspect_labels.detach().tolist()
        opinion_labs = opinion_labels.detach().tolist()

        tsne_dict['sentiment_vecs'] += sentiment_encs
        tsne_dict['aspect_vecs'] += aspect_encs
        tsne_dict['opinion_vecs'] += opinion_encs
        tsne_dict['sentiment_labels'] += sentiment_labs
        tsne_dict['aspect_labels'] += aspect_labs
        tsne_dict['opinion_labels'] += opinion_labs
        """

        # return original loss plus the characteristic-specific SCL losses
        loss = outputs[0] + opinion_contrastive_loss + sentiment_contrastive_loss + aspect_contrastive_loss
        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train_loss", loss)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        self.log('avg_train_loss', avg_train_loss)

    def validation_step(self, batch, batch_idx):
        loss, pred_outputs = self._step(batch)
        self.log('val_batch_loss', loss)
        return {"val_batch_loss": loss}
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_batch_loss"] for x in outputs]).mean()
        print("val_loss:\t", avg_loss )
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        cont_model = self.cont_model
        op_model = self.op_model
        as_model = self.as_model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in cont_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in op_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in as_model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.callback_metrics['avg_loss']), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, sents, task):
    """
    Compute scores given the predictions and gold labels and dump to file
    """
    device = torch.device(DEVICE)
    model.model.to(device)

    model.eval()
    model.model.eval()

    outputs, targets = [], []
    for batch in tqdm(data_loader):

        outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=args.max_seq_length*2,
                                    num_beams=args.num_beams)

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        outputs.extend(dec)
        targets.extend(target)

    scores, all_labels, all_preds = compute_scores(outputs, targets, task, False)
    results = {'labels_correct': all_labels, 'labels_pred': all_preds, 'output_pred': outputs, 'output_correct': targets, 'utterances': sents}
    ex_list = []

    for idx in range(len(all_preds)):
        new_dict = {}
        for key in results:
            new_dict[key] = results[key][idx]
        ex_list.append(new_dict)
    
    results = {'performance_metrics': scores, 'examples': ex_list}

    json.dump(results, open(f"{args.output_dir}/results-{args.dataset}.json", 'w'), indent=2, sort_keys=True)
    return scores

    
# check for top-level environment
if __name__ == '__main__':
    # initialization
    args = init_args()
    seed_everything(args.seed, workers=True)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(['[SSEP]'])




    # Get example from the train set
    dataset = GenSCLNatDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                        data_type='train', max_len=args.max_seq_length, task=args.task, truncate=args.truncate)
    data_sample = dataset[0]

    # sanity check
    # show one sample to check the code and the expected output format are correct
    print(f"Here is an example (from the train set):")
    print('Input :', tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True))
    print(data_sample['source_ids'])
    print('Output:', tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True))
    print(data_sample['target_ids'])

    # training process
    if args.do_train:
        print("\n****** Conducting Training ******")

        # initialize the T5 model
        tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        tfm_model.resize_token_embeddings(len(tokenizer))
        # initialize characteristic-specific representation models
        cont_model = LinearModel(args.model_name_or_path)
        op_model = LinearModel(args.model_name_or_path)
        as_model = LinearModel(args.model_name_or_path)
        cat_model = LinearModel(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer, cont_model, op_model, as_model, cat_model)

        if args.early_stopping:
            checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=args.output_dir, monitor='val_loss', mode='min', save_top_k=1
            )
            callback_list = [checkpoint_callback, LoggingCallback(), EarlyStopping(monitor="val_loss", mode='min', patience=3)]
        else:
            callback_list = [LoggingCallback()]

        # prepare trainer args
        train_params = dict(
            default_root_dir=args.output_dir,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            gpus=args.n_gpu,
            gradient_clip_val=1.0,
            max_epochs=args.num_train_epochs,
            auto_lr_find=False,
            deterministic=True,
            #auto_scale_batch_size=True,
            #callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode='min'), LoggingCallback()],
            callbacks=callback_list
        )
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)

        if args.early_stopping:
            ex_weights = torch.load(checkpoint_callback.best_model_path)['state_dict']
            model.load_state_dict(ex_weights)
            
        model.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        print("Finish training and saving the model!")

    # evaluation
    if args.do_direct_eval:
        print("\n****** Conduct Evaluating with the last state ******")

        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

        print()
        test_dataset = GenSCLNatDataset(tokenizer, data_dir=args.dataset, 
                                data_type='test', max_len=args.max_seq_length, task=args.task, truncate=args.truncate)
        test_loader = DataLoader(test_dataset, args.eval_batch_size, num_workers=4)

        # compute the performance scores
        evaluate(test_loader, model, test_dataset.sentence_strings, args.task)

    if args.do_inference:
        print("\n****** Conduct inference on trained checkpoint ******")

        # initialize the T5 model from previous checkpoint
        model_path = args.model_name_or_path
        print(f"Loading trained model from {model_path}")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        tfm_model = T5ForConditionalGeneration.from_pretrained(model_path)

        # representations are only used during loss calculation
        cont_model = LinearModel(args.model_name_or_path)
        op_model = LinearModel(args.model_name_or_path)
        as_model = LinearModel(args.model_name_or_path)
        cat_model = LinearModel(args.model_name_or_path)
        model = T5FineTuner(args, tfm_model, tokenizer, cont_model, op_model, as_model, cat_model)

        sents, _ = read_line_examples_from_file(f'data/{args.dataset}/test.txt')

        print()
        test_dataset = GenSCLNatDataset(tokenizer, data_dir=args.dataset, 
                                data_type='test', max_len=args.max_seq_length, task=args.task, truncate=args.truncate)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4)

        # compute the performance scores
        evaluate(test_loader, model, test_dataset.sentence_strings, args.task)
    