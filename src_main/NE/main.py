import time
import warnings
warnings.filterwarnings("ignore")
import json
import torch
from transformers import DebertaV2Tokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

from util import form_dataset, id2tag
from model import NE_Model
from eval import eval_ne


def adjust_learning_rate_exp(optimizer, epoch, num_ne_epochs, ne_lr, qg_lr, exp_num=1, decay_every=20):
    if epoch < num_ne_epochs:
        lr = ne_lr * (exp_num ** (epoch // decay_every))
    else:
        lr = qg_lr * (exp_num ** (epoch // decay_every))
    if hasattr(optimizer, 'module'):
        for param_group in optimizer.module.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    print("adjusted lr to {}".format(lr))


def build_optimizer_and_scheduler(model, model_config):
    # Optimizer with layer-wise lr-decay in BERT encoder
    # Scheduler with warm-up & linear lr-decay

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    bert_lr, head_lr, layer_lr_decay_rate = model_config['bert_lr'], model_config['head_lr'], model_config.get('layer_lr_decay_rate', 1)
    
    # Layer-wise lr-decay for bert
    bert_layer_lr, bert_num_layers = {}, model.bert_encoder.config.num_hidden_layers
    for layer_ix in range(bert_num_layers):
        bert_layer_lr['.layer.' + str(bert_num_layers-layer_ix-1) + '.'] = bert_lr * (layer_lr_decay_rate ** layer_ix)
    bert_layer_lr['.embeddings.'] = bert_lr * (layer_lr_decay_rate ** bert_num_layers)
    
    # Set optimizer lr & weight_decay
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    for name, param in parameters:
        # Different lr scheduling for pretrained & random init params
        if 'bert_encoder' in name:
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


def train_ne(model, train_loader, dev_loader, test_loader, model_config, do_eval_only=False):
    
    best_metrics = 0.
    
    output_model_file = model_config['checkpoint_route']
    latest_model_file = output_model_file[:(output_model_file.rfind('.pth'))] + "_latest.pth"
    log_output_file = model_config['log_dir'] + 'log_'+time.ctime()[4:19].replace(':','.')+'.dat'

    device_ids = [int(gpu_id) for gpu_id in model_config.get('gpus', '0,1').split(',')]
    print("Running on gpu:", device_ids)
    main_gpu = device_ids[0]
    device = torch.device("cuda:{}".format(main_gpu) if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if model_config.get('original_ne', True):
        # optimizer defined after transfer to GPU
        ne_lr_enc = model_config['ne_lr_enc']
        weight_decay = model_config.get('weight_decay', 0)
        print('weight decay:', weight_decay)
        # optimizer = torch.optim.SGD(model.decoder.parameters(), lr=0.0001, momentum=0.8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=ne_lr_enc, weight_decay=weight_decay)     # 4e-6
        scheduler = None
    else:
        model_config['train_loader'], model_config['dev_loader'] = train_loader, dev_loader
        optimizer, scheduler = build_optimizer_and_scheduler(model, model_config)

    # print every how many batches?
    print_every = 10
    lr_decay_exp, decay_every = model_config['lr_decay_exp'], model_config['decay_every']

    for epoch in range(3000):
        if do_eval_only:
            best_metrics = eval_ne(model, model_config, dev_loader, device, model_config['tokenizer'], log_output_file,\
                                        output_model_file, epoch, best_metrics, save_res=False)
            eval_ne(model, model_config, test_loader, device, model_config['tokenizer'], log_output_file,\
                                        output_model_file, epoch, 1.0, save_res=False)
            exit(0)

        if model_config.get('original_ne', True):
            # adjust lr every epoch
            adjust_learning_rate_exp(optimizer, epoch, -1, None, ne_lr_enc, exp_num=lr_decay_exp, decay_every=decay_every)
        running_loss = 0.

        for batch_i, cur_item in enumerate(train_loader):

            quess, masks, tags = cur_item
            quess, masks, tags = quess.to(device), masks.to(device), tags.to(device)
            
            optimizer.zero_grad()
 
            loss = model(enc_input_ids=quess, enc_attention_mask=masks, ne_tags=tags)
            loss.backward()
            optimizer.step()
            if not (scheduler is None):
                scheduler.step()
            running_loss += loss.item()

            if batch_i % print_every == print_every-1:  # print every 20 mini-batches
                f = open(log_output_file, 'a', encoding='utf-8')
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, batch_i + 1, running_loss / print_every))
                f.write('[%d, %5d] loss: %.5f\n' %
                      (epoch + 1, batch_i + 1, running_loss / print_every))
                f.close()
                running_loss = 0.
        
        # evaluate every epoch
        best_metrics = eval_ne(model, model_config, dev_loader, device, model_config['tokenizer'], log_output_file,\
                                output_model_file, epoch, best_metrics, save_res=True)
        # save latest checkpoint every 20 epochs
        if epoch % 20 == 19:
            torch.save(model.state_dict(), output_model_file)


def main(model_config, do_eval_only=False):
    
    # Dataloaders
    if len(id2tag) == 3:
        train_full_fn = '../data/NEQG_gold/Train_NE_gold.json'
        test_full_fn = '../data/NEQG_gold/Test_NE_gold.json'
    else:
        train_full_fn = '../data/NEQG_gold/Train_NE_gold_w_tag.json'
        test_full_fn = '../data/NEQG_gold/Test_NE_gold_w_tag.json'
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    train_batch, test_batch = model_config['train_batch'], model_config['test_batch']

    # Get train-dev split
    split_fn = '../data/meta/train_dev_split.json'
    with open(split_fn) as fin_split:
        split_dat = json.load(fin_split)
    train_loader, dev_loader, test_loader = form_dataset(train_full_fn, test_full_fn, tokenizer_dir, tokenizer_class, train_batch, test_batch, split_dat)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)
    model_config['tokenizer'] = tokenizer
    
    model_config['loss_function'] = torch.nn.CrossEntropyLoss()
    load_chkpt = not model_config['load_pre']
    # initiate a new model
    model = NE_Model(model_config)
    if load_chkpt or do_eval_only:
        # load checkpoints
        output_model_file = model_config['checkpoint_route']
        pretrain_model_file = model_config.get('pretrain_checkpoint_route', output_model_file)
        print("loading: ", pretrain_model_file)
        model.load_state_dict(torch.load(pretrain_model_file, map_location=torch.device('cpu')))
    
    train_ne(model, train_loader, dev_loader, test_loader, model_config, do_eval_only=do_eval_only)


if __name__ == '__main__':
    config_fn = '../data/checkpoint/NE_Roberta_w_tag/config.json'
    with open(config_fn) as fin:
        model_config = json.load(fin)
    main(model_config, do_eval_only=True)
