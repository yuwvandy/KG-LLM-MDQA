import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    #general setting
    parser.add_argument("--dataset", type=str, default="reason")
    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--local-rank", type=int)

    #model setting
    parser.add_argument("--max_source_text_len", type=int, default=512)
    parser.add_argument("--max_target_text_len", type=int, default=512)


    parser.add_argument("--source_text", type=str, default="input")
    parser.add_argument("--target_text", type=str, default="output")

    
    #training setting
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--train_bsz", type=int, default=4)
    parser.add_argument("--eval_bsz", type=int, default=4)
    parser.add_argument("--train_epochs", type=float, default=3)
    parser.add_argument("--val_epochs", type=float, default=1)
    
    parser.add_argument("--lr", type=float, default=1e-4)
    
    parser.add_argument("--model",
                        default="t5-large", type=str)
    
    
    return parser.parse_args()