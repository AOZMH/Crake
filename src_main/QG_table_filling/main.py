import sys
sys.path.append('../')
import time
import warnings
warnings.filterwarnings("ignore")
import json
from inspect import signature
import torch
if '_use_new_zipfile_serialization' in str(signature(torch.save)):
    serialize_arg = {'_use_new_zipfile_serialization': False}
else:
    serialize_arg = {}
from transformers import DebertaV2Tokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

from QG_table_filling.util import form_table_filling_dataset
from QG_table_filling.model import Table_Filling_Model
from QG_table_filling.eval import eval_neqg_tf
from NE.model import NE_Model


def build_dataloaders_and_models_qg(model_config):
    
    # Dataloaders & tokenizer
    train_qg_fn = '../data/NEQG_gold/Train_QG_gold.json'
    test_qg_fn = '../data/NEQG_gold/Test_QG_gold.json'
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    train_batch, test_batch = model_config['train_batch'], model_config['test_batch']
    # Get train-dev split
    split_fn = '../data/meta/train_dev_split.json'
    with open(split_fn) as fin_split:
        split_dat = json.load(fin_split)
    
    train_loader, dev_loader, test_loader = form_table_filling_dataset(train_qg_fn, test_qg_fn, tokenizer_dir, tokenizer_class, train_batch, test_batch, split_dat)
    model_config['train_loader'], model_config['dev_loader'], model_config['test_loader'] = train_loader, dev_loader, test_loader
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    model_config['tokenizer'] = tokenizer

    # Model
    load_chkpt = not model_config['load_pre'] or model_config['do_eval_only']
    # initiate a new model
    model = Table_Filling_Model(model_config)
    if load_chkpt:
        # load checkpoints
        output_model_file = model_config['checkpoint_route']
        pretrain_model_file = model_config.get('pretrain_checkpoint_route', output_model_file)
        print("loading: ", pretrain_model_file)
        model.load_state_dict(torch.load(pretrain_model_file, map_location=torch.device('cpu')))
    model_config['model'] = model
    
    # Load an NE model for evaluation, in case of no NEQG co-training
    # 如果只训练QG模型(无联训), 评估时需要一个NE模型, 这里可以指定一个训练好的
    if 'ne_model_config_route' in model_config.keys():
        ne_model_config_fn = model_config['ne_model_config_route']
        print("Loading assisting NE model from {}...".format(ne_model_config_fn))
        with open(ne_model_config_fn) as fin:
            ne_model_config = json.load(fin)
        ne_model = NE_Model(ne_model_config)
        ne_model.load_state_dict(torch.load(ne_model_config['checkpoint_route'], map_location=torch.device('cpu')))
        print('Assisting NE model loaded from {}.'.format(ne_model_config['checkpoint_route']))
        model_config['ne_gpus'] = ne_model_config['gpus']
    else:
        print('No assisting NE model specified.')
        model_config['ne_gpus'] = model_config['gpus']
        #assert(False)
        ne_model = None
    model_config['ne_model'] = ne_model

    return model_config


def build_optimizer_and_scheduler(model, model_config):
    # Optimizer with layer-wise lr-decay in BERT encoder
    # Scheduler with warm-up & linear lr-decay

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    bert_lr, head_lr, layer_lr_decay_rate = model_config['bert_lr'], model_config['head_lr'], model_config.get('layer_lr_decay_rate', 1)
    
    # Layer-wise lr-decay for bert
    bert_layer_lr, bert_num_layers = {}, model.encoder.bert_encoder.config.num_hidden_layers
    for layer_ix in range(bert_num_layers):
        bert_layer_lr['.layer.' + str(bert_num_layers-layer_ix-1) + '.'] = bert_lr * (layer_lr_decay_rate ** layer_ix)
    bert_layer_lr['.embeddings.'] = bert_lr * (layer_lr_decay_rate ** bert_num_layers)
    
    # Set optimizer lr & weight_decay
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    for name, param in parameters:
        # Different lr scheduling for pretrained & random init params
        if 'encoder.bert_encoder' in name:
            base_lr = bert_lr
            # Apply layer-wise decay
            for bert_layer_name, lr in bert_layer_lr.items():
                if bert_layer_name in name:
                    base_lr = lr
                    break
        else:
            base_lr = head_lr
        
        # Avoid weight_decay for LayerNorm & bias
        cur_weight_decay = 0.0 if any(item in name for item in no_decay) else model_config['weight_decay']

        #print('{}\t{}\t{}'.format(name, base_lr, cur_weight_decay))
        params = {'params': [param], 'lr': base_lr, 'weight_decay': cur_weight_decay}
        optimizer_grouped_parameters.append(params)
    
    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(model_config.get('adam_beta1', 0.9), model_config.get('adam_beta2', 0.999)),
                      lr=model_config['head_lr'],
                      eps=model_config.get('adam_eps', 1e-8),
                      weight_decay=model_config['weight_decay'],
                      #correct_bias=False
    ) # AdamW construction
    
    # Construct scheduler
    total_train_steps = len(model_config['train_loader']) * model_config.get('total_epochs', 1000)
    num_warmup_steps = int(model_config.get('warmup_rate', 0.2) * total_train_steps)
    print('Warmup / Total steps = {} / {}'.format(num_warmup_steps, total_train_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_train_steps)
    return optimizer, scheduler


def train_tfqg(model_config):
    # Train a table-filling QG model
    do_eval_only = model_config['do_eval_only']
    best_actual_acc = 0.

    output_model_file = model_config['checkpoint_route']
    latest_model_file = output_model_file[:(output_model_file.rfind('.pth'))] + "_latest.pth"
    log_output_file = model_config['log_dir'] + 'log_'+time.ctime()[4:19].replace(':','.')+'.dat'

    # Load dataloaders & init models
    model_config = build_dataloaders_and_models_qg(model_config)
    
    dec_total_params = sum(p.numel() for p in model_config['model'].parameters()) - sum(p.numel() for p in model_config['model'].encoder.bert_encoder.parameters())
    ne_total_params = sum(p.numel() for p in model_config['ne_model'].ne_classifier.parameters())
    print('ne params: {}, gc_dec params: {}, total dec params:{}'.format(ne_total_params, dec_total_params, ne_total_params + dec_total_params))
    #exit(0)

    model = model_config['model']
    train_loader, dev_loader, test_loader = model_config['train_loader'], model_config['dev_loader'], model_config['test_loader']

    device_ids = [int(gpu_id) for gpu_id in model_config.get('gpus', '0,1').split(',')]
    ne_device_ids = [int(gpu_id) for gpu_id in model_config.get('ne_gpus', '0,1').split(',')]
    main_gpu = device_ids[0]
    ne_gpu = ne_device_ids[-1]
    use_ne_cache = model_config.get('use_ne_cache', False)
    device = torch.device("cuda:{}".format(main_gpu) if torch.cuda.is_available() else "cpu")
    ne_device = torch.device("cuda:{}".format(ne_gpu) if torch.cuda.is_available() and (not use_ne_cache) else "cpu")
    print("QG model running on gpu:", device)
    model = model.to(device)
    if model_config['ne_model']:
        print("NE model running on gpu:", ne_device)
        model_config['ne_model'] = model_config['ne_model'].to(ne_device)
    
    # Construct optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(model, model_config)

    # Training
    print_every = 10
    running_loss_1, running_loss_2 = 0., 0.
    for epoch in range(model_config.get('total_epochs', 1000)):
        if do_eval_only:
            cur_total_acc, cur_actual_acc = eval_neqg_tf(model, dev_loader, model_config['tokenizer'], ne_model=model_config['ne_model'], remove_homo_vars=model_config.get('remove_homo_vars', False), use_ne_cache=use_ne_cache)
            with open(log_output_file, 'a') as fin:
                msg = "Dev Total Acc: {}, Dev Actual Acc: {}.".format(cur_total_acc, cur_actual_acc)
                fin.write(msg + '\n')
                print(msg)

            cur_total_acc, cur_actual_acc = eval_neqg_tf(model, test_loader, model_config['tokenizer'], ne_model=model_config['ne_model'], remove_homo_vars=model_config.get('remove_homo_vars', False), use_ne_cache=use_ne_cache)
            with open(log_output_file, 'a') as fin:
                msg = "Test Total Acc: {}, Test Actual Acc: {}.".format(cur_total_acc, cur_actual_acc)
                fin.write(msg + '\n')
                print(msg)
            exit(0)

        for batch_i, cur_item in enumerate(train_loader):
            running_loss_1, running_loss_2 = 0., 0.
            
            quess, masks, adj_matrices, matrix_masks, node_types, node_ptrs, node_ends = cur_item
            quess, masks, adj_matrices, matrix_masks = quess.to(device), masks.to(device), adj_matrices.to(device), matrix_masks.to(device)
            
            # for decoder, input is [0,-2] and output is [1,-1]
            nodes_in, nodes_out = \
                node_types[:,:-1].clone().detach(), node_types[:, 1:].clone().detach()

            optimizer.zero_grad()
 
            all_loss = model(
                enc_input_ids = quess,
                enc_attention_mask = masks,
                adj_matrices = adj_matrices,
                matrix_masks = matrix_masks,
            )
            
            # weighted-sum two losses
            matrix_label_loss, symmetric_loss = all_loss
            loss = matrix_label_loss + symmetric_loss * model_config.get('symm_loss_ratio', 1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss_1 += matrix_label_loss.item()
            running_loss_2 += symmetric_loss.item()

            if batch_i % print_every == print_every-1:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.5f %.5f' %
                        (epoch + 1, batch_i + 1, running_loss_1 / print_every, running_loss_2 / print_every))
                if not model_config.get('use_nni', False):
                    with open(log_output_file, 'a') as fin:
                        fin.write('[%d, %5d] loss: %.5f %.5f\n' %
                            (epoch + 1, batch_i + 1, running_loss_1 / print_every, running_loss_2 / print_every))
                running_loss_1, running_loss_2 = 0., 0.
        
        #if epoch < 10:
        #    continue
        # evaluate every epoch
        cur_total_acc, cur_actual_acc = eval_neqg_tf(model, dev_loader, model_config['tokenizer'], ne_model=model_config['ne_model'], remove_homo_vars=model_config.get('remove_homo_vars', False), use_ne_cache=use_ne_cache)
        msg = "Total Acc: {}, Actual Acc: {}.".format(cur_total_acc, cur_actual_acc)
        msg += '\nBest Actual Acc: {}'.format(max(cur_actual_acc, best_actual_acc))
        print(msg)
        with open(log_output_file, 'a') as fin:
            fin.write(msg + '\n')
        
        if cur_actual_acc > best_actual_acc:
            best_actual_acc = cur_actual_acc
            if model_config.get('save_res', True) and (not do_eval_only):
                print('Saving model...')
                torch.save(model.state_dict(), output_model_file, **serialize_arg)
        # save latest checkpoint every 20 epochs
        #if epoch % 20 == 19:
        #    torch.save(model.state_dict(), latest_model_file, _use_new_zipfile_serialization = False)
        
    # Report final results
    torch.save(model.state_dict(), output_model_file, _use_new_zipfile_serialization = False)


if __name__ == '__main__':
    config_fn = '../data/checkpoint/QG_TF_Roberta_base/config.json'
    with open(config_fn) as fin:
        model_config = json.load(fin)
        model_config['do_eval_only'] = True
    train_tfqg(model_config)
