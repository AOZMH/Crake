import os
import time
import warnings
warnings.filterwarnings("ignore")
import json
import torch
from transformers import DebertaV2Tokenizer, RobertaTokenizer

from util import form_dataset
from model import RE_Model
from eval import eval_re


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


def train_re(model, train_loader, dev_loader, test_loader, model_config, do_eval_only=False):
    
    best_mrr, test_best_mrr = 0., 0.
    
    output_model_file = model_config['checkpoint_route']
    latest_model_file = output_model_file[:(output_model_file.rfind('.pth'))] + "_latest.pth"
    log_output_file = model_config['log_dir'] + 'log_'+time.ctime()[4:19].replace(':','.')+'.dat'
    
    # determine gpu ids and parallel model
    os.environ['CUDA_VISIBLE_DEVICES'] = model_config.get('gpus', '0,1')
    main_gpu = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    print('Running on gpu:', os.environ['CUDA_VISIBLE_DEVICES'])
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # optimizer defined after transfer to GPU
    re_lr_enc = model_config['re_lr_enc']
    weight_decay = model_config.get('weight_decay', 0)
    print('weight decay:', weight_decay)
    if hasattr(model, 'module'):
        optimizer = torch.optim.AdamW(model.module.parameters(), lr=re_lr_enc, weight_decay=weight_decay)     # 4e-6
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=re_lr_enc, weight_decay=weight_decay)     # 4e-6

    # print/eval every how many batches?
    print_every, eval_every, save_latest_every = 20, 400, 400
    topk = model_config.get('pred_topk', 5)
    lr_decay_exp, decay_every = model_config['lr_decay_exp'], model_config['decay_every']

    epoch, batch_i = 0,0
    if do_eval_only:
        print("Do eval only.\nDev:")
        eval_re(model, dev_loader, log_output_file, output_model_file,\
                                epoch, best_mrr, topk=topk,\
                                ranges=model_config['dev_triple_ranges'], v_set=model_config['raw_dev_set'],\
                                save_res=False, output_full_result=False)
        print('\nTest:')
        eval_re(model, test_loader, log_output_file, output_model_file,\
                                epoch, test_best_mrr, topk=topk,\
                                ranges=model_config['test_triple_ranges'], v_set=model_config['raw_test_set'],\
                                save_res=False, output_full_result=True)
        exit(0)

    running_loss, correct, total = 0.,0,0
    # adjust lr every epoch
    adjust_learning_rate_exp(optimizer, epoch, 0, re_lr_enc, re_lr_enc, exp_num=lr_decay_exp, decay_every=decay_every)
    train_iter = iter(train_loader)
    while True:
        try:
            itm = train_iter.next()
            batch_i += 1
        except:
            # show train acc every epoch
            print('Train acc: %.8f' % (correct/total))
            # adjust lr every epoch
            epoch += 1
            if epoch >= 20:
                exit(0)
            adjust_learning_rate_exp(optimizer, epoch, 0, re_lr_enc, re_lr_enc, exp_num=lr_decay_exp, decay_every=decay_every)
            batch_i, running_loss, correct, total = 0,0.,0,0
            train_iter = iter(train_loader)
            itm = train_iter.next()
        
        sents, type_masks, att_masks, tags = itm
        sents, type_masks, att_masks, tags = sents.cuda(), type_masks.cuda(), att_masks.cuda(), tags.cuda()

        optimizer.zero_grad()
        loss = model(sents,\
                enc_attention_mask=att_masks,\
                token_type_ids=type_masks,\
                re_tags=tags)
        
        #loss = lossf(re_logits, tags)
        loss = loss.mean()      # support both single & multi-gpu training
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        #correct += (torch.max(re_logits, 1)[1] == tags).sum().item()
        total += tags.shape[0]

        if batch_i % print_every == print_every-1:  # print every 20 mini-batches
            f = open(log_output_file, 'a', encoding='utf-8')
            print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, batch_i + 1, running_loss / print_every))
            f.write('[%d, %5d] loss: %.5f\n' %
                    (epoch + 1, batch_i + 1, running_loss / print_every))
            f.close()
            running_loss = 0.
    
        # evaluate every epoch
        if batch_i % eval_every == (eval_every-1):
            best_mrr = eval_re(model, dev_loader, log_output_file, output_model_file,\
                                epoch, best_mrr, topk=topk,\
                                ranges=model_config['dev_triple_ranges'], v_set=model_config['raw_dev_set'],\
                                save_res=True, output_full_result=False)
        
        # save latest checkpoint every 20 epochs
        if batch_i % save_latest_every == (save_latest_every-1):
            print("Saving latest model...")
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), latest_model_file)
            else:
                torch.save(model.state_dict(), latest_model_file)


def main(model_config, do_eval_only=False):
    
    # Dataloaders
    train_full_fn = '../data/RE_gold/Train_RE_gold_preprocessed_1hop_split.json'
    dev_full_fn = '../data/RE_gold/Dev_RE_gold_preprocessed_1hop_split.json'
    test_full_fn = '../data/RE_gold/Test_RE_gold_preprocessed_1hop_split.json'
    tokenizer_dir = '../data/pretrained/roberta_large_tokenizer/'
    tokenizer_class = RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_dir)

    train_batch, test_batch = model_config['train_batch'], model_config['test_batch']
    num_workers = model_config.get('loader_num_workers', 0)
    sample_ratio = model_config.get('sample_ratio', 50)

    train_loader, dev_loader, test_loader, dev_triple_ranges, raw_dev_set, test_triple_ranges, raw_test_set = form_dataset(train_full_fn, dev_full_fn, test_full_fn, train_batch, test_batch, tokenizer, num_workers, sample_ratio)
    model_config['tokenizer'] = tokenizer
    model_config['dev_triple_ranges'] = dev_triple_ranges
    model_config['raw_dev_set'] = raw_dev_set
    model_config['test_triple_ranges'] = test_triple_ranges
    model_config['raw_test_set'] = raw_test_set
    
    model_config['loss_function'] = torch.nn.CrossEntropyLoss()
    load_chkpt = not model_config['load_pre']
    # initiate a new model
    model = RE_Model(model_config)
    if load_chkpt or do_eval_only:
        # load checkpoints
        output_model_file = model_config['checkpoint_route']
        pretrain_model_file = model_config.get('pretrain_checkpoint_route', output_model_file)
        print("loading: ", pretrain_model_file)
        model.load_state_dict(torch.load(pretrain_model_file, map_location=torch.device('cpu')))
    
    train_re(model, train_loader, dev_loader, test_loader, model_config, do_eval_only=do_eval_only)


if __name__ == '__main__':
    config_fn = '../data/checkpoint/RE_Roberta_small_samp/config.json'
    with open(config_fn) as fin:
        model_config = json.load(fin)
    main(model_config, do_eval_only=False)
