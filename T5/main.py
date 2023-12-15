import torch
from dataset import load_dataset, dataset_process
from utils import seed_everything
import os
from parse import parse_args
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from learn import train, eval
import math
import pandas as pd
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

def run(train_data, val_data, model, tokenizer, args):
    train_set = dataset_process(
        train_data,
        tokenizer,
        args.max_source_text_len,
        args.max_target_text_len,
        args.source_text,
        args.target_text,
    )

    val_set = dataset_process(
        val_data,
        tokenizer,
        args.max_source_text_len,
        args.max_target_text_len,
        args.source_text,
        args.target_text,
    )

    train_params = {
        "batch_size": args.train_bsz,
        "shuffle": False,
        "num_workers": args.num_workers,
    }

    val_params = {
        "batch_size": args.eval_bsz,
        "shuffle": False,
        "num_workers": args.num_workers,
    }

    train_loader = DataLoader(train_set, **train_params, sampler = DistributedSampler(train_set))
    val_loader = DataLoader(val_set, **val_params, sampler = DistributedSampler(val_set))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    best_loss = math.inf
    
    for epoch in range(args.train_epochs):
        loss = train(epoch, tokenizer, model, train_loader, optimizer)

        if loss < best_loss:
            best_loss = loss

            model.module.save_pretrained("./model/{}_{}".format(args.dataset, args.model))
            tokenizer.save_pretrained("./model/{}_{}".format(args.dataset, args.model))

            print("Epoch: {}, Loss: {}".format(epoch, loss))

        if epoch % args.val_epochs == 0:
            predictions, labels, questions = eval(tokenizer, model, val_loader)
    
    final_df = pd.DataFrame({'Questions': questions, "Generated Text": predictions, "Actual Text": labels})

    final_df.to_csv("./res/{}_{}/predictions.csv".format(args.dataset, args.model))




if __name__ == "__main__":
    args = parse_args()
    args.path = os.getcwd()

    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    seed_everything(args.seed)
    train_data, val_data = load_dataset(args) 

    # for inference
    # tokenizer = T5Tokenizer.from_pretrained("./model/{}".format(args.dataset))
    # model = T5ForConditionalGeneration.from_pretrained("./model/{}".format(args.dataset))
    
    # for training
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).cuda()
    model = DistributedDataParallel(model)

    run(train_data, val_data, model, tokenizer, args)
