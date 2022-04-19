# Table-filling-based QG model

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class Multitask_Encoder(nn.Module):

    def __init__(self, load_pre=True, ne_label_size=15, re_label_size=2, pretrain_route='./data/pretrained', use_lstm=False, pretrain_config_fn='./data/pretrained/config.json'):
        
        super(Multitask_Encoder, self).__init__()
        if load_pre:
            print('Loading pretrained weights from {}'.format(pretrain_route))
            self.bert_encoder = AutoModel.from_pretrained('roberta-large')
        else:
            print('Random initializing model. Config from {}'.format(pretrain_config_fn))
            my_config = AutoConfig.from_pretrained(pretrain_config_fn)
            self.bert_encoder = AutoModel.from_config(my_config)
            #assert(False)
        
        bert_dim = self.bert_encoder.config.hidden_size
        if use_lstm:
            self.lstm = nn.LSTM(bert_dim, bert_dim // 2, batch_first=True,
                            num_layers=2, bidirectional=True)
        else:
            self.lstm = None

        self.dropout = nn.Dropout(self.bert_encoder.config.hidden_dropout_prob)
        # classifier head for NE
        if ne_label_size == -1:
            print("No NE layer built.")
            self.ne_classifier = None
        else:
            print("Built with NE layer.")
            self.ne_classifier = nn.Linear(bert_dim, ne_label_size)
        # classifier head for RE
        if re_label_size == -1:
            print("No RE layer built.")
            self.re_classifier = None
        else:
            print("Built with RE layer.")
            self.re_classifier = nn.Linear(bert_dim, re_label_size)
        
    def forward(self, enc_input_ids, enc_attention_mask=None, token_type_ids=None):

        outputs = self.bert_encoder(enc_input_ids,
                            attention_mask=enc_attention_mask,
                            token_type_ids=token_type_ids)

        # encoder output: <batch, seq_len_src, bert_hid_dim(=768)>
        enc_output = outputs[0]
        enc_output = self.dropout(enc_output)        

        if self.lstm is not None:
            enc_output = self.lstm(enc_output)[0]
            enc_output = self.dropout(enc_output)

        ret_list = [enc_output, None, None]
        if type(self.ne_classifier) != type(None):
            # ne_logits: <batch, seq_len_src, ne_tag_size>
            ne_logits = self.ne_classifier(enc_output)
            ret_list[1] = ne_logits
        if type(self.re_classifier) != type(None):
            # cls output: <batch, bert_hid_dim(=768)>
            cls_output = enc_output[:, 0, :]
            cls_output = self.dropout(cls_output)
            # re_logits: <batch, re_tag_size>
            re_logits = self.re_classifier(cls_output)
            ret_list[2] = re_logits

        return ret_list


class Table_Filling_Model(nn.Module):

    def __init__(self, model_config):
        super(Table_Filling_Model, self).__init__()
        # Resolve params for encoder
        load_pre = model_config.get('load_pre', True)
        ne_label_size = model_config.get('ne_label_size', 15)
        re_label_size = model_config.get('re_label_size', 2)
        pretrain_route = model_config.get('pretrain_route', './data/pretrained') 
        use_lstm = model_config.get('use_lstm', False)
        dropout_prob = model_config.get('dropout_prob', 0.2)
        logit_dropout_prob = model_config.get('logit_dropout_prob', 0.2)
        activation = model_config.get('activation', 'gelu')
        pretrain_config_fn = model_config.get('pretrain_config_fn', './data/pretrained/config.json')
        self.gama, self.alpha = model_config.get('gama', 1), model_config.get('alpha', 1)
        self.beta, self.theta = model_config.get('beta', 1), model_config.get('theta', 1)
        biaff_linear_dim = model_config.get('biaff_linear_dim', 256)
        # Since only predict if an edge exists, qg_tag_size = 2
        qg_tag_size = model_config.get('qg_tag_size', 2)
        
        # Encoder (bert and possible NE & RE head)
        self.encoder = Multitask_Encoder(load_pre=load_pre, ne_label_size=ne_label_size,\
                                            re_label_size=re_label_size, pretrain_route=pretrain_route,\
                                            use_lstm=use_lstm, pretrain_config_fn=pretrain_config_fn)
        hid_dim = self.encoder.bert_encoder.config.hidden_size
        
        # Biaff-head
        self.activation_func = getattr(nn.functional, activation)
        self.fc_head = nn.Linear(hid_dim, biaff_linear_dim)
        self.fc_tail = nn.Linear(hid_dim, biaff_linear_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        if logit_dropout_prob > 0:
            self.logit_dropout = nn.Dropout(p=logit_dropout_prob)
        else:
            self.logit_dropout = lambda x : x
        # TODO 这里randn好还是zero好?
        # U <tag_size, biaff_linear_dim + 1, biaff_linear_dim + 1>
        self.biaff_U = nn.Parameter(torch.randn(qg_tag_size, biaff_linear_dim + 1, biaff_linear_dim + 1))
        self.label_loss = nn.CrossEntropyLoss()

        # Prediction mode, to be compatible with NE, QG... evaluation routines
        self.predicion_mode = 'NEQG'
    
    def fc_pass(self, inp_feature, fc_layer):
        # FC wrapper to include: dropout, activation (gelu), layer_norm...
        fc_output = self.dropout(self.activation_func(fc_layer(inp_feature)))
        return fc_output
    
    def add_ones_to_last_column(self, fc_output):
        # Add extra row with only ONEs for Biaff calculation
        return torch.cat([fc_output, torch.ones(fc_output.shape[:-1] + (1,), device=self.biaff_U.device)], dim=-1)
    
    def forward(self, enc_input_ids, enc_attention_mask=None, token_type_ids=None,\
                adj_matrices=None, matrix_masks=None, ne_tags=None):
        # Params containing:
        #   enc_input_ids       <batch, sent_len> (max sent length in the batch)
        #   enc_attention_mask  <batch, sent_len>
        #   adj_matrices        <batch, sent_len, sent_len>
        #   matrix_masks        <batch, sent_len, sent_len>, bool tensor
        #   ne_tags             <batch, snet_len>, long tensor
        
        # Bert encoder output
        # enc_output <batch, sent_len, bert_hid_dim>
        enc_output, ne_logits, re_logits = self.encoder(enc_input_ids,
                            enc_attention_mask=enc_attention_mask,
                            token_type_ids=token_type_ids)
        
        # Biaff operations:
        # 1. MLPs (layernorm, dropout, gelu...) -> <batch, sent_len, biaff_linear_dim>
        # 2. Add extra row to head/tail_fc_res -> <batch, sent_len, biaff_linear_dim + 1>
        head_fc_res = self.fc_pass(enc_output, self.fc_head)
        head_fc_res = self.add_ones_to_last_column(head_fc_res)
        tail_fc_res = self.fc_pass(enc_output, self.fc_tail)
        tail_fc_res = self.add_ones_to_last_column(tail_fc_res)
        # 3. Use einsum to cal Biaff(h_i, h_j)
        # -> biaff_scores <batch, sent_len, sent_len, qg_tag_size(=2)>
        biaff_scores = torch.einsum('bxi, lij, byj -> bxyl', head_fc_res, self.biaff_U, tail_fc_res)
        
        all_loss = []
        if adj_matrices is not None:
            # At training phase, calculate losses
            # Logit dropout
            logit_dropout_res = self.logit_dropout(biaff_scores)
            # Matrix label loss. Since matrix masks is Bool:
            #   logit_dropout_res[matrix_masks] -> <unmasked_item_num, qg_tag_size>
            #   adj_matrices[matrix_masks] -> <unmasked_item_num>
            matrix_label_loss = self.label_loss(logit_dropout_res[matrix_masks], adj_matrices[matrix_masks])
            # Symmetric loss
            sm_mask_scores = torch.softmax(biaff_scores, dim=-1) * matrix_masks.unsqueeze(dim=-1).float()
            symmetric_loss = torch.abs(sm_mask_scores - sm_mask_scores.transpose(1, 2)).sum(dim=-1)[matrix_masks].mean()
            
            # Add loss to output
            all_loss = [matrix_label_loss, symmetric_loss]

        if ne_tags is not None:
            # NE tag loss
            assert(ne_logits is not None)
            dropout_ne_logits = self.logit_dropout(ne_logits)
            ne_logits_flat = dropout_ne_logits.view(-1, dropout_ne_logits.shape[-1])
            ne_tags_flat = ne_tags.view(-1)
            ne_tag_loss = self.label_loss(ne_logits_flat, ne_tags_flat)
            # Add loss to output
            all_loss.append(ne_tag_loss)

        if len(all_loss) != 0:
            return all_loss
        
        if self.predicion_mode == 'NEQG':
            # At test phase, simply return biaff_scores & leave matrix decoding outside the model
            sm_biaff_scores = torch.softmax(biaff_scores, dim=-1)
            return sm_biaff_scores, ne_logits
        elif self.predicion_mode == 'NE':
            # Only output ne_logits for NE evaluation
            return ne_logits

    def set_prediction_mode(self, target_mode):
        self.predicion_mode = target_mode
