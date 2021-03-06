import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_batch_size', default=64, type=int, )
parser.add_argument('--eval_batch_size', default=64, type=int, )
parser.add_argument('--test_batch_size', default=64, type=int, )
parser.add_argument('--learning_rate', default=3e-5, type=float, )
parser.add_argument('--num_train_epochs', default=3, type=int, )
parser.add_argument('--max_seq_len', default=100, type=int, )
parser.add_argument('--seed', default=42, type=int, )
parser.add_argument('--fgm', action='store_true' )
parser.add_argument('--pretrain',action='store_true' )
parser.add_argument('--do_predict', action='store_true' )
parser.add_argument('--warmup_steps', default=0.1, type=float, )
parser.add_argument('--logging_steps', default=1000, type=int, )
parser.add_argument('--model_type', default='bert', type=str, )
parser.add_argument('--model_path', default='mac_bert', type=str, )
parser.add_argument('--save_path', default='mac_bert', type=str, )
args = parser.parse_args()
