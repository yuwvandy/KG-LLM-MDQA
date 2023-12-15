import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    #general setting
    parser.add_argument("--dataset", type=str, default="MuSiQue", choices=["HotpotQA", 'IIRC', '2WikiMQA', 'MuSiQue'])
    parser.add_argument("--seed", type=int, default=1028)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--local_rank", type=str, default=1)

    #model setting
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--max_q_sp_len", type=int, default=350)
    parser.add_argument("--max_q_len", type=int, default=70)
    
    #training setting
    parser.add_argument("--num_workers", type=int, default=30)
    parser.add_argument("--train_bsz", type=int, default=150)
    parser.add_argument("--eval_bsz", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=50)
    parser.add_argument("--top_k", type=int, default=15)
    
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warm_ratio", default=0.1, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--weight_decay", default=0.0, type=float)
    
    #eval setting
    parser.add_argument("--do_predict", default=True,
                        action='store_true', help="Whether to run eval on the dev set.")
    
    parser.add_argument("--model_name",
                        default="bert-base-uncased", type=str)
    parser.add_argument("--max_grad_norm",
                        default=2.0, type=float)
    
    
    return parser.parse_args()