import torch
import random
import numpy as np
import  pandas as pd
from sklearn.metrics import f1_score
from functools import partial
import scipy as sp
from codem.configm.config import  args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True




def get_data(data_path,datatype):
    query=[]
    candidate=[]
    label=[]
    text_id=[]
    if datatype=='train':
        with open(data_path) as f:
            for i in f:
                dict_txt=eval(i)
                if dict_txt['query']=='':
                    continue
                for j in dict_txt['candidate']:
                    if j['text']=='':
                        continue
                    query.append(dict_txt['query'])
                    candidate.append(j['text'])
                    label.append(j['label'])

        data=pd.DataFrame({'query':query,'candidate':candidate,'label':label})

    else:
        with open(data_path) as f:
            for i in f:
                dict_txt=eval(i)

                for j in dict_txt['candidate']:

                    text_id.append(dict_txt['text_id'])
                    query.append(dict_txt['query'])
                    candidate.append(j['text'])


        data=pd.DataFrame({'text_id':text_id,'query':query,'candidate':candidate})

    return data


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = [1., 1., 1.]

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        X_p = np.argmax(X_p * coef, axis=1)
        y_t = y
        ll = f1_score(y_t, X_p, average='macro')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        if type(self.coef_) is list:
            initial_coef = self.coef_
        else:
            initial_coef = self.coef_['x']
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='Nelder-Mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        X_p = X_p * coef

        return X_p

    def coefficients(self):
        return self.coef_['x']

def dynamic_batch(batch):



    input_ids,attention_mask,token_type_ids=[],[],[]
    label=[]
    collate_max_len = 0


    for sample in batch:

        collate_max_len=max(collate_max_len,len(sample['input_ids']))


    for sample in batch:

        length = len(sample['input_ids'])

        input_ids.append(sample['input_ids'] + [0] * (collate_max_len - length))
        attention_mask.append(sample['attention_mask']+ [0] * (collate_max_len - length))
        token_type_ids.append(sample['token_type_ids']+[0]*(collate_max_len-length))
        if 'label' in sample and not args.pretrain:
            label.append(sample['label'])
        elif 'label' in sample and args.pretrain:
            label.append(sample['label']+[0]*(collate_max_len-length))


    input_ids = torch.tensor(input_ids).long()
    attention_mask = torch.tensor(attention_mask).long()
    token_type_ids=torch.tensor(token_type_ids).long()
    if label!=[]:
        label = torch.tensor(label).long()
        return  {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label':label,

        }
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'label': label,

    }


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.5, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

