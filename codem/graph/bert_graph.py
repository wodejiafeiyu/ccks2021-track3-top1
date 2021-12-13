from transformers import BertPreTrainedModel,BertModel
import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss

class bert_classify(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config, )
        config.output_hidden_states = True
        self.num_labels =config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)
        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,


        )

        hidden_layers = outputs[2]
        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        outputs = (logits,) + outputs[2:]
        if labels is not None:

            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss=loss1

            outputs = (loss.mean(),) + outputs

        return outputs



