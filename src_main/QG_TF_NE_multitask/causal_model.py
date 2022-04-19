# Causal multitasking model for NE & QG_TF

import torch
import torch.nn as nn

# 复用QG中的encoder, 在上面加上table process即可
from QG_table_filling.model import Multitask_Encoder


class Private_Encoder(nn.Module):
    # A private encoder for each task
    # Can read hidden states & label embs from upstream tasks for causal learning
    def __init__(self, model_config):
        super(Private_Encoder, self).__init__()
        # Resolve input channels to decide input dimension
        print('Building a private encoder with:', model_config['input_channels'])
        input_dim, hid_dim = model_config['bert_dim'], model_config['private_hid_dim']
        if model_config['input_channels']['upstream_hidden']:
            input_dim += hid_dim
        if model_config['input_channels']['upstream_label_emb']:
            input_dim += model_config['label_emb_dim']
        
        # Build private_encoder model
        self.factorizer = nn.Linear(input_dim, hid_dim)
        self.private_encoder_arch = model_config.get('private_encoder_arch', 'transformer')

        if self.private_encoder_arch == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=model_config['private_nheads'], activation=model_config.get('activation', 'gelu'), dim_feedforward=model_config.get('private_transformer_feed_forward_dim', 1024))
            self.private_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_config['private_nlayers'])
        elif self.private_encoder_arch == 'bilstm':
            self.lstm_encoder = nn.LSTM(input_size=hid_dim, hidden_size=hid_dim // 2, num_layers=model_config['private_nlayers'], bidirectional=True)
            self.private_encoder = lambda x : self.lstm_encoder(x)[0]
        elif self.private_encoder_arch == 'fcn':
            self.private_encoder = lambda x : x
        else:
            raise NotImplementedError
        # Assiting blocks
        self.dropout = nn.Dropout(p=model_config.get('dropout_prob', 0.2))
        self.activation_func = getattr(nn.functional, model_config.get('activation', 'gelu'))
    
    def forward(self, enc_output, upstream_hidden=None, upstream_label_emb=None):
        # Concat channels
        private_input = enc_output
        if upstream_hidden is not None:
            private_input = torch.cat([private_input, upstream_hidden], dim=-1)
        if upstream_label_emb is not None:
            private_input = torch.cat([private_input, upstream_label_emb], dim=-1)
        # Factorize to hidden dimension
        private_hidden = self.dropout(self.activation_func(self.factorizer(private_input)))
        # Private encoding
        private_hidden = private_hidden.transpose(0,1)  # batch_first align
        private_encoder_output = self.private_encoder(private_hidden)
        private_encoder_output = private_encoder_output.transpose(0,1)
        if self.private_encoder_arch != 'fcn':
            # Activate & dropout before feed to task-specific heads
            private_encoder_output = self.dropout(self.activation_func(private_encoder_output))
        return private_encoder_output


class Label_Encoder(nn.Module):
    # Generate label encodings given task label logits
    # enc_mode
    #   warmup: Embed gold labels
    #   train:  Gumble-softmax on ne_logits
    #   test:   One-hot ne_logits
    def __init__(self, model_config):
        super(Label_Encoder, self).__init__()
        self.enc_mode = 'warmup'
        self.label_size = model_config['ne_label_size']
        self.tau = model_config.get('gumble_tau', 0.05)
        print('Gumble softmax tau:', self.tau)
        self.label_embedding = nn.Linear(self.label_size, model_config['label_emb_dim'], bias=False)
    
    def forward(self, label_logits, gold_labels=None):
        # label_logits: <batch, len, num_labels>
        # gold_labels:  <batch, len>, long tensor
        if self.enc_mode == 'warmup':
            emb_input = nn.functional.one_hot(gold_labels, num_classes=self.label_size).float()
        elif self.enc_mode == 'train':
            emb_input = nn.functional.gumbel_softmax(label_logits, tau=self.tau)
        elif self.enc_mode == 'test':
            pred_labels = label_logits.max(dim=-1).indices
            emb_input = nn.functional.one_hot(pred_labels, num_classes=self.label_size).float()
        emb_output = self.label_embedding(emb_input)
        return emb_output


class Causal_NE_QGTF_Model(nn.Module):

    def __init__(self, model_config):
        super(Causal_NE_QGTF_Model, self).__init__()
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
        hid_dim = model_config['private_hid_dim']
        # Since only predict if an edge exists, qg_tag_size = 2
        qg_tag_size = model_config.get('qg_tag_size', 2)
        ne_tag_size = model_config['ne_label_size']
        
        # Encoder (bert and possible NE & RE head)
        # Causal-mtl models need to build NE, QG layers outside encoder
        self.encoder = Multitask_Encoder(load_pre=load_pre, ne_label_size=-1,\
                                            re_label_size=-1, pretrain_route=pretrain_route,\
                                            use_lstm=use_lstm, pretrain_config_fn=pretrain_config_fn)
        model_config['bert_dim'] = self.encoder.bert_encoder.config.hidden_size

        # Private encoder for NE & QG_TF
        model_config['input_channels'] = {'upstream_hidden': False, 'upstream_label_emb': False}
        self.ne_private_encoder = Private_Encoder(model_config)
        model_config['input_channels'] = {'upstream_hidden': model_config['transfer_hidden'], 'upstream_label_emb': model_config['transfer_label']}
        self.qg_input_channels = model_config['input_channels']
        self.qg_tf_private_encoder = Private_Encoder(model_config)

        # NE head
        self.ne_classifier = nn.Linear(hid_dim, ne_tag_size)
        
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

        if model_config['transfer_label']:
            # Label embedding when utilizing upstream labels in downstream encodings
            self.ne_label_encoder = Label_Encoder(model_config)

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
        
        # NE encoder & head
        ne_private_enc_output = self.ne_private_encoder(enc_output)
        ne_logits = self.ne_classifier(ne_private_enc_output)
        # NE label encoder
        if self.qg_input_channels['upstream_label_emb']:
            ne_label_enc_output = self.ne_label_encoder(ne_logits, ne_tags)
        # QG_TF encoder
        additional_channels = {
            'upstream_hidden': ne_private_enc_output if self.qg_input_channels['upstream_hidden'] else None,
            'upstream_label_emb': ne_label_enc_output if self.qg_input_channels['upstream_label_emb'] else None,
        }
        qg_tf_private_enc_output = self.qg_tf_private_encoder(enc_output, **additional_channels)
        
        # (QG_TF head) Biaff operations:
        # 1. MLPs (layernorm, dropout, gelu...) -> <batch, sent_len, biaff_linear_dim>
        # 2. Add extra row to head/tail_fc_res -> <batch, sent_len, biaff_linear_dim + 1>
        head_fc_res = self.fc_pass(qg_tf_private_enc_output, self.fc_head)
        head_fc_res = self.add_ones_to_last_column(head_fc_res)
        tail_fc_res = self.fc_pass(qg_tf_private_enc_output, self.fc_tail)
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
    
    def set_ne_label_encoder_mode(self, target_mode):
        if not hasattr(self, 'ne_label_encoder'):
            print('Trying to adjust label encoder mode without label encoder, nothing will happen. Please remove "label_enc_warmup_epochs" key in config.')
        else:
            self.ne_label_encoder.enc_mode = target_mode
