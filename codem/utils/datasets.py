import numpy as np
import  collections
import  random

class simdatasets(object):

    def __init__(self, query,candidate,
                 label,tokenizer,
                 max_seq_len,pretrain=False):
        
        self.query=query
        self.candidate=candidate
        self.label=label
        self.tokenizer=tokenizer
        self.max_seq_len = max_seq_len
        self.pretrain=pretrain
        self.MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

    def __len__(self):
        return len(self.query)

    def __getitem__(self, item):
        
        if self.pretrain:
            
            tokenize_result = self.tokenizer.encode_plus(self.query[item],
                                                       self.candidate[item],
                                             max_length=self.max_seq_len,
                                             truncation=True,
                                             truncation_strategy='longest_first', )

            if self.label[item]!=-1:
                tokenize_result['input_ids'] = tokenize_result['input_ids'] + [self.label[item]+1] + [102]

                tokenize_result['attention_mask'] = tokenize_result['attention_mask'] + [1] + [1]
                tokenize_result['token_type_ids'] = tokenize_result['token_type_ids'] + [0] + [0]



            input_ids, labels = self._mask_tokens(tokenize_result['input_ids'])

            return {
                'input_ids': input_ids,
                'attention_mask': tokenize_result['attention_mask'],
                'token_type_ids': tokenize_result['token_type_ids'],
                'label':labels}



        else:
            
            tokenize_result=self.tokenizer.encode_plus(self.query[item],
                                                       self.candidate[item],
                                      max_length=self.max_seq_len,
                                      truncation=True,
                                      truncation_strategy='longest_first', )


            if self.label[item] !=None:
    
                return {
                    'input_ids': tokenize_result["input_ids"],
                    'attention_mask': tokenize_result["attention_mask"],
                    'token_type_ids': tokenize_result["token_type_ids"],
                    'label': self.label[item]
    
                }

            return {
                'input_ids': tokenize_result["input_ids"],
                'attention_mask': tokenize_result["attention_mask"],
                'token_type_ids': tokenize_result["token_type_ids"],
    
            }

    def _mask_tokens(self, inputs) :

        def single_mask_tokens(tokens, max_ngram=3):

            ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
            pvals = 1. / np.arange(1, max_ngram + 1)
            pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
            cand_indices = []
            for (i, token) in enumerate(tokens):
                if token == 101 or token == 102:
                    continue
                cand_indices.append(i)

            num_to_mask =  max(1, int(round(len(tokens) * 0.15)))
            random.shuffle(cand_indices)  #
            masked_token_labels = []
            covered_indices = set()

            for index in cand_indices:

                n = np.random.choice(ngrams, p=pvals)
                if len(masked_token_labels) >= num_to_mask:
                    break
                if index in covered_indices:
                    continue
                if index < len(cand_indices) - (n - 1):
                    for i in range(n):
                        ind = index + i
                        if ind in covered_indices:
                            continue
                        covered_indices.add(ind)
                        # 80% of the time, replace with [MASK]
                        if random.random() < 0.8:
                            masked_token = 103

                        else:
                            # 10% of the time, keep original
                            if random.random() < 0.5:
                                masked_token = tokens[ind]
                            # 10% of the time, replace with random word
                            else:
                                masked_token = random.choice(range(0,self.tokenizer.vocab_size))
                        masked_token_labels.append(self.MaskedLmInstance(index=ind, label=tokens[ind]))
                        tokens[ind] = masked_token


            masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)

            target = len(tokens) * [-100]
            for p in masked_token_labels:
                target[p.index] = p.label

            return tokens, target

        a_mask_tokens,ta=single_mask_tokens(inputs)

        return a_mask_tokens,ta

