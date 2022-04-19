import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class NE_Model(nn.Module):
    # NE model, i.e. transformer encoder + seq-labeling head

    def __init__(self, model_config):
        
        super(NE_Model, self).__init__()
        load_pre = model_config.get('load_pre', True)
        ne_label_size = model_config.get('ne_label_size', 3)
        pretrain_route = model_config.get('pretrain_route', '../data/pretrained/deberta_xlarge_model')
        use_lstm = model_config.get('use_lstm', False)
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
        if use_lstm:
            self.lstm = nn.LSTM(bert_dim, bert_dim // 2, batch_first=True,
                            num_layers=2, bidirectional=True)
        else:
            self.lstm = None

        self.dropout = nn.Dropout(self.bert_encoder.config.hidden_dropout_prob)
        # classifier head for NE
        self.ne_classifier = nn.Linear(bert_dim, ne_label_size)
        
    def forward(self, enc_input_ids, enc_attention_mask=None, token_type_ids=None, ne_tags=None):

        outputs = self.bert_encoder(enc_input_ids,
                            attention_mask=enc_attention_mask,
                            token_type_ids=token_type_ids)

        # encoder output: <batch, seq_len_src, bert_hid_dim(=768)>
        enc_output = outputs[0]     # last_hidden_state
        enc_output = self.dropout(enc_output)

        if self.lstm is not None:
            enc_output = self.lstm(enc_output)[0]
            enc_output = self.dropout(enc_output)

        # ne_logits: <batch, seq_len_src, ne_tag_size>
        ne_logits = self.ne_classifier(enc_output)

        if type(ne_tags) != type(None):
            ne_logits_flat = ne_logits.view(-1, ne_logits.shape[-1])
            ne_tags_flat = ne_tags.view(-1)
            loss = self.lossf(ne_logits_flat, ne_tags_flat)
            return loss

        return enc_output, ne_logits
