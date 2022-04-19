# Definition & Wrapper of all models

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class RE_Model(nn.Module):
    # RE model, i.e. transformer encoder + cls head

    def __init__(self, model_config):
        
        super(RE_Model, self).__init__()
        load_pre = model_config.get('load_pre', True)
        re_label_size = model_config.get('re_label_size', 2)
        pretrain_route = model_config.get('pretrain_route', '../data/pretrained/deberta_xlarge_model')
        pretrain_config_fn = model_config.get('pretrain_config_fn', '../data/pretrained/deberta_xlarge_model/config.json')
        self.lossf = model_config.get('loss_function', None)

        if load_pre:
            print('Loading pretrained weights from {}'.format(pretrain_route))
            self.bert_encoder = AutoModel.from_pretrained(pretrain_route)
        else:
            print('Random initializing model. Config from {}'.format(pretrain_config_fn))
            my_config = AutoConfig.from_pretrained(pretrain_config_fn)
            self.bert_encoder = AutoModel.from_config(my_config)
        
        bert_dim = self.bert_encoder.config.hidden_size
        hidden_dropout_prob = model_config.get('hidden_dropout_prob', self.bert_encoder.config.hidden_dropout_prob)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        # classifier head for RE
        self.re_classifier = nn.Linear(bert_dim, re_label_size)
        
    def forward(self, enc_input_ids, enc_attention_mask=None, token_type_ids=None, re_tags=None):

        # NOTE: 这里 token_type_ids 不需要, 因为Roberta不用这个输入, 否则会报错 (https://github.com/huggingface/transformers/issues/1443)
        if 'RobertaModel' in self.bert_encoder.config.architectures:
            token_type_ids = None
        outputs = self.bert_encoder(enc_input_ids,
                            attention_mask=enc_attention_mask,
                            token_type_ids=token_type_ids)

        # cls output: <batch, bert_hid_dim(=768)>
        cls_output = outputs[1]     # pooler_output; old versions of transformers does not have named dict!
        cls_output = self.dropout(cls_output)

        # re_logits: <batch, re_tag_size>
        re_logits = self.re_classifier(cls_output)

        if type(re_tags) != type(None):
            loss = self.lossf(re_logits, re_tags)
            return loss

        return -1, re_logits  # Add a dummy return to accord with server.py
