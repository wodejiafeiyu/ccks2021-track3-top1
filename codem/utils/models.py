import torch
import os

from sklearn.metrics import f1_score
from codem.configm.config import args
from torch.utils.data import DataLoader, RandomSampler
from .datasets import  simdatasets
from .tools import dynamic_batch,FGM
from transformers import BertTokenizer
from transformers import AdamW
from transformers import  get_linear_schedule_with_warmup
from tqdm import  tqdm
import numpy as np
from codem.graph import  bert_graph,nezha_graph,utils
from torch.cuda.amp import autocast as autocast
import shutil
from transformers import  BertConfig,BertForMaskedLM

class simmodels(object):
    def __init__(self,model_path,save_path,model_type='bert'):
        if not args.do_predict:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.copy(f'{model_path}/config.json',f'{save_path}/config.json')
            shutil.copy(f'{model_path}/vocab.txt',f'{save_path}/vocab.txt')


        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if 'bert' in model_type and not args.pretrain:
            bert_config = BertConfig.from_pretrained(f'{model_path}/config.json')
            bert_config.num_labels = 3

            model=bert_graph.bert_classify.from_pretrained(f"{model_path}/pytorch_model.bin",config=bert_config)
        elif 'nezha' in model_type and not args.pretrain:
            nezhaconfig = nezha_graph.nezhaconfig.from_json_file(f'{model_path}/config.json')
            model = nezha_graph.nezha_classify(nezhaconfig, num_labels=3)
            utils.torch_init_model(model, model_path + '/pytorch_model.bin')

        elif 'bert' in model_type and args.pretrain:

            bert_config = BertConfig.from_pretrained(f'{model_path}/config.json')

            model = BertForMaskedLM.from_pretrained(f'{model_path}/pytorch_model.bin',
                                                    config=bert_config)

        else :
            nezhaconfig = nezha_graph.nezhaconfig.from_json_file(f'{model_path}/config.json')
            model = nezha_graph.BertForMaskedLM(nezhaconfig)
            utils.torch_init_model(model, model_path + '/pytorch_model.bin')




        self.model=model
        self.model.to(self.device)
        self.toknizer=BertTokenizer.from_pretrained(model_path)
        self.save_path=save_path

    def train(self,train_data,dev_data=None):
        
        train_dataset=simdatasets(train_data['query'],
                                  train_data['candidate'],
                                  train_data['label'],
                                  self.toknizer,args.max_seq_len,pretrain=args.pretrain)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=args.train_batch_size,
                                      collate_fn=dynamic_batch)

        t_total = len(train_dataloader) * args.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01, 'lr': args.learning_rate},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps * t_total, num_training_steps=t_total
        )

        global_step = 1
        logging_loss =  10000
        best_score = 0.0

        self.model.zero_grad()
        for num in range(args.num_train_epochs):
            print("----- Epoch {}/{} ;-----".format(num + 1, args.num_train_epochs))
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", )
            epoch_loss=0.0
            for step, batch in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                self.model.train()
                inputs = {
                    "input_ids": batch['input_ids'].to(self.device),
                    "attention_mask": batch['attention_mask'].to(self.device),
                    "token_type_ids": batch['token_type_ids'].to(self.device),
                    "labels": batch['label'].to(self.device),
                }
                if args.pretrain:
                    with autocast():
                        outputs = self.model(**inputs)
                        loss = outputs[0]
                else:
                    outputs = self.model(**inputs)
                    loss = outputs[0]

                loss.backward()

                epoch_loss += loss.item()
                if args.fgm:
                    fgm = FGM(self.model)
                    fgm.attack()
                    output1 = self.model(**inputs)
                    loss_adv = output1[0]

                    loss_adv.backward()
                    fgm.restore()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                global_step += 1



                # 保存验证集上最好的参数
                if global_step % args.logging_steps == 0 and  not args.pretrain:

                    cur_score = self.evaluate(dev_data)
                    print("currently!! best_score is{}, cur_score is "
                          "{}".format(best_score, cur_score))

                    if cur_score >= best_score:
                        best_score = cur_score
                        torch.save(self.model.state_dict(), self.save_path + '/pytorch_model.bin')
                        tqdm.write('saving_model')

            if  args.pretrain:
                tqdm.write(str(epoch_loss))
            if epoch_loss <= logging_loss and args.pretrain :
                logging_loss = epoch_loss
                torch.save(self.model.state_dict(), self.save_path + '/pytorch_model.bin')
                tqdm.write('saving_model')

        return best_score

    
    def evaluate(self,eval_data):
        eval_dataset=simdatasets(eval_data['query'],
                                  eval_data['candidate'],
                                  eval_data['label'],
                                  self.toknizer,args.max_seq_len)

        eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=dynamic_batch)
        pre_label_all = []
        label_true = []

        for batch in eval_dataloader:
            self.model.eval()

            with torch.no_grad():
                inputs = {
                    "input_ids": batch['input_ids'].to(self.device),
                    "attention_mask": batch['attention_mask'].to(self.device),
                    "token_type_ids": batch['token_type_ids'].to(self.device),

                }
                label_true.extend(batch['label'].detach().cpu().numpy())
                outputs = self.model(**inputs)

                pre_label_all.append(outputs[0].softmax(-1).detach().cpu().numpy())

        pre_label_all = np.concatenate(pre_label_all)

        return f1_score(label_true, np.argmax(pre_label_all, axis=-1), average='macro')

    def predict(self,test_data):
        test_dataset=simdatasets(test_data['query'],
                                  test_data['candidate'],
                                  [None]*len(test_data['query']),
                                  self.toknizer,args.max_seq_len)

        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=dynamic_batch)
        pre_label_all = []

        for batch in tqdm(test_dataloader,'infer...'):
            self.model.eval()
            with torch.no_grad():
                inputs = {
                    "input_ids": batch['input_ids'].to(self.device),
                    "attention_mask": batch['attention_mask'].to(self.device),
                    "token_type_ids": batch['token_type_ids'].to(self.device),
                }

                outputs = self.model(**inputs)
                pre_label_all.append(outputs[0].softmax(-1).detach().cpu().numpy())
        pre_label_all = np.concatenate(pre_label_all)

        return pre_label_all