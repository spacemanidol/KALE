""" Align the embeddings of noised data"""
import argparse
import json
import pdb
import logging
import copy
import math
import os
import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import (
    AdamW,
    set_seed,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import numpy as np
from sparseml.pytorch.optim import ScheduledModifierManager, ScheduledOptimizer
from sparseml.pytorch.utils import export_onnx
from sparseml.transformers.sparsification import Trainer
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Compress Model by Aliging the embeddings")
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="If you want to train",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="If you want to train",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="The name of the train file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="loss temperature",
    )
    parser.add_argument(
        "--kldiv",
        action="store_true",
        help="If passed, will use a kldiv instead of cosine distance.",
    )
    parser.add_argument(
        "--margin",
        action="store_true",
        help="If passed, will use margin loss. Control the impact of margin loss by using temperature",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        help="The name of the dev file.",
    )
    parser.add_argument(
        "--cosine_margin",
        type=float,
        default=0.2,
        help="cosine loss"
    )
    parser.add_argument(
        "--target_model",
        type=str,
        default=None,
        help="Use to create a separate target to align on",
    )
    parser.add_argument(
        "--add_pooler_target",
        action="store_true",
        help="attach a poler to the end of target model"
    )
    parser.add_argument(
        "--pooler_input_target",
        default=768,
        type=int,
        help="Dimension input for pooler target model"
    )
    parser.add_argument(
        "--pooler_output_orig",
        default=768,
        type=int,
        help="Dimension output pooler original model",
    )
    parser.add_argument(
        "--add_pooler",
        action="store_true",
        help="attach a poler to the end target model"
    )
    parser.add_argument(
        "--pooler_input",
        default=768,
        type=int,
        help="Dimension input for pooler target model"
    )
    parser.add_argument(
        "--pooler_output",
        default=768,
        type=int,
        help="Dimension output pooler target model",
    )
    parser.add_argument(
        "--positive_temp",
        type=float,
        default=0.6,
        help="positive temperature for cosine loss"
    )
    parser.add_argument(
        "--negative_temp",
        type=float,
        default=0.2,
        help="positive temperature for cosine loss"
    )
    parser.add_argument(
        "--margin_temp",
        type=float,
        default=1.0,
        help="control the importance of margin ranking loss between negative and positive variants of the query",
    )
    parser.add_argument(
        "--mse_loss",
        action="store_true",
        help="If passed MSE loss will be used on cosine distance"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='spacemanidol/esci-all-distilbert-base-uncased-5e-5',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the  Tokenizers library).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="train in mixed precision"
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help='Recipe for compression of model',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="random Seed",
   )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--onnx_name",
        type=str,
        default='model.onnx',
        help="onnx model name",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--hard_negatives",
        action="store_true",
        help="Use hard negatives instead of random negatives",
    )
    parser.add_argument(
        "--struct_prune",
        action="store_true",
        help="structural prune",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        help="data parralel",
    )
    parser.add_argument(
        "--layers",
        default=12,
        type=int,
        help="layers to keep on model",
    )
    parser.add_argument(
        "--negatives",
        default=2,
        type=int,
        help='negatives to use',
    )

    args = parser.parse_args()
    return args

def struct_prune(model, num_layers_to_keep):
    oldModuleList = model.encoder.layer #s
    newModuleList = nn.ModuleList()
    layers = {1:[0], 2:[0,11],3:[0,5,11],6:[0,2,4,6,8,11], 9:[0,1,3,4,5,6,7,9,11]}
    layers = {1:[0], 2:[0,5],3:[0,2,5]}
    #layers = {1:[0], 2:[0,2]}
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[layers[num_layers_to_keep][i]])
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList
    print(copyOfModel)
    return copyOfModel

def main():
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print(args)
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
    set_seed(args.seed)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    data_files = {   }
    if args.do_train:
        data_files["train"] = args.train_file
    if args.do_eval:
        data_files["dev"] = args.dev_file
    if args.do_eval == False and args.do_train == False:
        print("Model is neither training nor evaluating")
        exit(-1)
    raw_datasets = load_dataset("json", data_files=data_files)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.target_model == None:
        target_model = copy.deepcopy(model)
    else:
        target_config = AutoConfig.from_pretrained(args.target_model)
        target_model = AutoModel.from_pretrained(
            args.target_model,
            from_tf=bool(".ckpt" in args.target_model),
            config=target_config,
        )
    target_model  = target_model.to(device)
    target_model.eval()
    if args.struct_prune != None:
        if args.layers != 12:
            model = struct_prune(model,args.layers)
    
    padding = "max_length"
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['query_text'], max_length=args.max_length, padding=padding, truncation=True)
        negative_targets_input = tokenizer(examples['negative'], max_length=args.max_length, padding=padding, truncation=True)
        hard_negative_targets_input = tokenizer(examples['hard_negative'], max_length=args.max_length, padding=padding, truncation=True)
        hard_negative_target_inputs = [tokenizer(examples['hard_negatives'][i], max_length=args.max_length, padding=padding, truncation=True) for i in range(args.negatives)]
        model_inputs["negative_input_ids"] = negative_targets_input["input_ids"]
        model_inputs["hard_negative_input_ids"] = hard_negative_targets_input["input_ids"]
        model_inputs["negative_attention_mask"] = negative_targets_input["attention_mask"]
        model_inputs["hard_negative_attention_mask"] = hard_negative_targets_input["attention_mask"]
        return model_inputs
    if args.do_eval and args.do_train == False:
        processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets['dev'].column_names,
        )
    else:
        processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets['train'].column_names,
        )
    if args.do_train:
        train_dataset = processed_datasets["train"]
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    if args.do_eval:
        eval_dataset = processed_datasets["dev"]
    
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size,drop_last =True
        )
    if args.do_eval:
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.batch_size,drop_last=True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if args.do_train:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        logger.info(f"Loading the recipe for compression")
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        if args.recipe != None:
            manager = ScheduledModifierManager.from_yaml(args.recipe)
            optimizer = manager.modify(model, optimizer, num_update_steps_per_epoch)
            logger.info(f"Recipe Loaded")

    if args.do_eval:
        num_eval_steps = len(eval_dataloader)

    total_batch_size = args.batch_size  * args.gradient_accumulation_steps    
    # Only show the progress bar once on each machine.
    completed_steps = 0
    cosine_embedding_loss_positive = torch.nn.CosineEmbeddingLoss(margin=0)
    kl_loss = nn.KLDivLoss(reduction="sum")
    mse_loss_fn = nn.MSELoss()
    cosine_embedding_loss_negative = torch.nn.CosineEmbeddingLoss(margin=args.cosine_margin)
    margin_ranking_loss = nn.MarginRankingLoss()
    cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
    if torch.cuda.device_count() > 1 and args.dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    target_positive = torch.ones(args.batch_size).to(device)
    target_negative = target_positive * -1
    target_negative = target_negative.to(device)
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        progress_bar = tqdm(range(args.max_train_steps))
        with torch.cuda.amp.autocast():
            for epoch in range(args.num_train_epochs):
                if args.do_eval:
                    model.eval()
                    eval_loss = []
                    logger.info("***** Running Evaluation *****")
                    for step, batch in enumerate(eval_dataloader):
                        positive_inputs = {'input_ids':batch['input_ids'].to(device), 'attention_mask':batch['attention_mask'].to(device)}
                        negative_inputs = {'input_ids':batch['negative_input_ids'].to(device), 'attention_mask':batch['negative_attention_mask'].to(device)}
                        with torch.no_grad():
                            positive = model(input_ids=positive_inputs['input_ids'],attention_mask=positive_inputs['attention_mask']).last_hidden_state[:,0]
                            negative = model(input_ids=negative_inputs['input_ids'],attention_mask=negative_inputs['attention_mask']).last_hidden_state[:,0]
                            anchored_positive = target_model(input_ids=positive_inputs['input_ids'],attention_mask=positive_inputs['attention_mask']).last_hidden_state[:,0]
                            anchored_negative = target_model(input_ids=negative_inputs['input_ids'],attention_mask=negative_inputs['attention_mask']).last_hidden_state[:,0]
                        if args.kldiv:
                            inputs = F.log_softmax(positive/ args.temperature, dim=-1)
                            targets = F.softmax(anchored_positive/ args.temperature, dim=-1)
                            loss_pos = kl_loss(inputs, targets) * (args.temperature ** 2) / (positive.numel() / positive.shape[-1])
                            inputs = F.log_softmax(negative/ args.temperature, dim=-1)
                            targets = F.softmax(anchored_negative/ args.temperature, dim=-1)
                            loss_neg = kl_loss(inputs, targets) * (args.temperature ** 2) / (negative.numel() / negative.shape[-1])
                        elif args.mse_loss:
                            candidates = torch.cosine_similarity(positive, anchored_positive)
                            loss_pos = mse_loss_fn(candidates,  torch.zeros(candidates.shape).to('cuda'))
                            candidates = torch.cosine_similarity(negative, anchored_negative)
                            loss_neg = mse_loss_fn(candidates,  torch.zeros(candidates.shape).to('cuda'))
                        else:
                            loss_pos = cosine_embedding_loss_positive(positive, anchored_positive,target_positive) * args.temperature
                            loss_neg = cosine_embedding_loss_negative(negative, anchored_negative,target_negative) * args.temperature
                        loss = (args.positive_temp * loss_pos) + (args.negative_temp * loss_neg)
                        if args.margin:
                            pos_sim = cos_sim(positive, anchored_positive)
                            neg_sim = cos_sim(negative, anchored_negative)
                            margin_loss = margin_ranking_loss(pos_sim, neg_sim, target_positive)
                            loss = loss + (args.margin_temp * margin_loss)
                        eval_loss.append(loss.item())
                    logger.info(f"Loss : {np.mean(eval_loss)}")
                model.train()
                for step, batch in enumerate(train_dataloader):
                    positive_inputs = {'input_ids':batch['input_ids'].to(device), 'attention_mask':batch['attention_mask'].to(device)}
                    negative_inputs = {'input_ids':batch['negative_input_ids'].to(device), 'attention_mask':batch['negative_attention_mask'].to(device)}
                    positive = model(input_ids=positive_inputs['input_ids'],attention_mask=positive_inputs['attention_mask']).last_hidden_state[:,0]
                    negative = model(input_ids=negative_inputs['input_ids'],attention_mask=negative_inputs['attention_mask']).last_hidden_state[:,0]
                    with torch.no_grad():
                        anchored_positive = target_model(input_ids=positive_inputs['input_ids'],attention_mask=positive_inputs['attention_mask']).last_hidden_state[:,0]
                        anchored_negative = target_model(input_ids=negative_inputs['input_ids'],attention_mask=negative_inputs['attention_mask']).last_hidden_state[:,0]
                    if args.kldiv:
                        inputs = F.log_softmax(positive/ args.temperature, dim=-1)
                        targets = F.softmax(anchored_positive/ args.temperature, dim=-1)
                        loss_pos = kl_loss(inputs, targets) * (args.temperature ** 2) / (positive.numel() / positive.shape[-1])
                        inputs = F.log_softmax(negative/ args.temperature, dim=-1)
                        targets = F.softmax(anchored_negative/ args.temperature, dim=-1)
                        loss_neg = kl_loss(inputs, targets) * (args.temperature ** 2) / (negative.numel() / negative.shape[-1])
                    elif args.mse_loss:
                        candidates = torch.cosine_similarity(positive, anchored_positive)
                        loss_pos = mse_loss_fn(candidates,  torch.zeros(candidates.shape).to('cuda'))
                        candidates = torch.cosine_similarity(negative, anchored_negative)
                        loss_neg = mse_loss_fn(candidates,  torch.zeros(candidates.shape).to('cuda'))
                    else:
                        loss_pos = cosine_embedding_loss_positive(positive, anchored_positive,target_positive) * args.temperature
                        loss_neg = cosine_embedding_loss_negative(negative, anchored_negative,target_negative) * args.temperature
                    loss = (args.positive_temp * loss_pos) + (args.negative_temp * loss_neg)
                    if args.margin:
                        pos_sim = cos_sim(positive, anchored_positive)
                        neg_sim = cos_sim(negative, anchored_negative)
                        margin_loss = margin_ranking_loss(pos_sim, neg_sim, target_positive)
                        loss = loss + (args.margin_temp * margin_loss)
                    if args.fp16:
                        scaler.scale(loss).backward()
                        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            scaler.step(optimizer)
                            lr_scheduler.step()
                            optimizer.zero_grad()
                            progress_bar.update(1)
                            completed_steps += 1
                            scaler.update()
                    else:
                        loss.backward()
                        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad()
                            progress_bar.update(1)
                            completed_steps += 1
                    if step % 100 == 0:
                        logger.info(f"Loss : {loss.item()}")
                    if isinstance(args.checkpointing_steps, int):
                        if completed_steps % args.checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            #if isinstance(model,type(nn.DataParallel(None))):
                            #    model.module.save_pretrained(args.output_dir)
                            #else:
                            model.save_pretrained(args.output_dir)
                    if completed_steps >= args.max_train_steps:
                        break
    if args.do_eval:
        model.eval()
        eval_loss = []
        logger.info("***** Running Evaluation *****")
        for step, batch in enumerate(eval_dataloader):
            positive_inputs = {'input_ids':batch['input_ids'].to(device), 'attention_mask':batch['attention_mask'].to(device)}
            negative_inputs = {'input_ids':batch['negative_input_ids'].to(device), 'attention_mask':batch['negative_attention_mask'].to(device)}
            with torch.no_grad():
                positive = model(input_ids=positive_inputs['input_ids'],attention_mask=positive_inputs['attention_mask']).last_hidden_state[:,0]
                negative = model(input_ids=negative_inputs['input_ids'],attention_mask=negative_inputs['attention_mask']).last_hidden_state[:,0]
                anchored_positive = target_model(input_ids=positive_inputs['input_ids'],attention_mask=positive_inputs['attention_mask']).last_hidden_state[:,0]
                anchored_negative = target_model(input_ids=negative_inputs['input_ids'],attention_mask=negative_inputs['attention_mask']).last_hidden_state[:,0]
            if args.kldiv:
                inputs = F.log_softmax(positive/ args.temperature, dim=-1)
                targets = F.softmax(anchored_positive/ args.temperature, dim=-1)
                loss_pos = kl_loss(inputs, targets) * (args.temperature ** 2) / (positive.numel() / positive.shape[-1])
                inputs = F.log_softmax(negative/ args.temperature, dim=-1)
                targets = F.softmax(anchored_negative/ args.temperature, dim=-1)
                loss_neg = kl_loss(inputs, targets) * (args.temperature ** 2) / (negative.numel() / negative.shape[-1])
            else:
                loss_pos = cosine_embedding_loss_positive(positive, anchored_positive,target_positive) * args.temperature
                loss_neg = cosine_embedding_loss_negative(negative, anchored_negative,target_negative) * args.temperature
            loss = (args.positive_temp * loss_pos) + (args.negative_temp * loss_neg)
            if args.margin:
                pos_sim = cos_sim(positive, anchored_positive)
                neg_sim = cos_sim(negative, anchored_negative)
                margin_loss = margin_ranking_loss(pos_sim, neg_sim, target_positive)
                loss = loss + (args.margin_temp * margin_loss)
            eval_loss.append(loss.item())
        logger.info(f"Loss : {np.mean(eval_loss)}")
    # create fake model input
    inputs = tokenizer(
        "Do you like kale?", return_tensors="pt", max_length=args.max_length, padding=padding, truncation=True
    )
    inputs = {'input_ids':inputs['input_ids'].to(device), 'attention_mask':inputs['attention_mask'].to(device)}
    onnx_file_path = os.path.join(args.output_dir, args.onnx_name)
    if args.dp:
        model.module.save_pretrained(args.output_dir)
        trainer = Trainer(
            model=target_model,
            model_state_path=args.output_dir,
            recipe=args.recipe,
            recipe_args=None,
            teacher=None,
        )
        applied = trainer.apply_manager(epoch=math.inf, checkpoint=None)
        export_onnx(model.module,inputs,onnx_file_path,convert_qat=True,)
    else:
        model.save_pretrained(args.output_dir)
        trainer = Trainer(
            model=model,
            model_state_path=args.output_dir,
            recipe=args.recipe,
            recipe_args=None,
            teacher=None,
        )
        applied = trainer.apply_manager(epoch=math.inf, checkpoint=None)
        export_onnx(model,inputs,onnx_file_path,convert_qat=True,)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

