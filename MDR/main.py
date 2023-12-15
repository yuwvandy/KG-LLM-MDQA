import os

from parse import parse_args
from dataset import load_dataset
from learn import train, eval
from utils import seed_everything
from loader import Dataset_process, Dataset_collate, Dataset_process2
from torch.utils.data import DataLoader
import math

import torch
from torch.optim import Adam
from transformers import (AutoConfig, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from model import Retriever

def run(train_data, val_data, model, tokenizer, collate, args):
    if args.do_train:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = Adam(optimizer_parameters, lr = args.lr, eps = args.adam_epsilon)
        
        if args.dataset == 'HotpotQA':
            train_dataset = Dataset_process(train_data, tokenizer, args, train = True)
        elif args.dataset == 'MuSiQue':
            train_dataset = Dataset_process2(train_data, tokenizer, args, train = True)

        train_dataloader = DataLoader(train_dataset, batch_size = args.train_bsz, pin_memory = True, collate_fn = collate, num_workers = args.num_workers, shuffle=True)

        if args.dataset == 'HotpotQA':
            val_dataset = Dataset_process(val_data, tokenizer, args, train = False)
        elif args.dataset == 'MuSiQue':
            val_dataset = Dataset_process2(val_data, tokenizer, args, train = False)

        val_dataloader = DataLoader(val_dataset, batch_size = args.eval_bsz, pin_memory = True, collate_fn = collate, num_workers = args.num_workers, shuffle=False)
        
        t_total = len(train_dataloader) * args.epochs
        warmup_steps = math.ceil(t_total * args.warm_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        best_mrr = 0

        for epoch in range(args.epochs):
            loss = train(model, train_dataloader, optimizer, scheduler, args)

            mrr_1, mrr_2 = eval(model, val_dataloader)
            mrr_avg = (mrr_1 + mrr_2) / 2

            if mrr_avg > best_mrr:
                best_mrr = mrr_avg
                torch.save(model.state_dict(), './model/{}/model.pt'.format(args.dataset))

                print("Epoch: {}, Loss: {}, MRR_1: {}, MRR_2: {}, Ave_MRR: {}".format(epoch, loss, mrr_1, mrr_2, (mrr_1 + mrr_2) / 2))
                  
    else:
        if args.dataset == 'HotpotQA':
            val_dataset = Dataset_process(val_data, tokenizer, args, train = False)
        elif args.dataset == 'MuSiQue':
            val_dataset = Dataset_process2(val_data, tokenizer, args, train = False)
        val_dataloader = DataLoader(val_dataset, batch_size = args.eval_bsz, pin_memory = True, collate_fn = collate, num_workers = args.num_workers, shuffle=False)
        
        mrr_1, mrr_2 = eval(model, val_dataloader)
        mrr_avg = (mrr_1 + mrr_2) / 2

        print("MRR_1: {}, MRR_2: {}, Ave_MRR: {}".format(mrr_1, mrr_2, (mrr_1 + mrr_2) / 2))
                

if __name__ == "__main__":
    args = parse_args()
    args.path = os.getcwd()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    seed_everything(args.seed)
    train_data, val_data = load_dataset(args.dataset) 

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    bert_config = AutoConfig.from_pretrained(args.model_name)

    model = Retriever(bert_config, args)

    if not args.do_train:
        model = load_saved(model, './model/{}/model.pt'.format(args.dataset), exact=False)

    model.to(args.device)
    model = torch.nn.DataParallel(model)

    run(train_data, val_data, model, tokenizer, Dataset_collate, args)