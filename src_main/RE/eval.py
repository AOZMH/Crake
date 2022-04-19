# RE evaluation routines: MRR, Top-k Recall
import json
import time
import torch


def show_mrr_and_topk_recall(logits, golds, topk, ranges, v_set, output_full_result):
    # ranges = [[l1, r1], [l2, r2], ...]
    #   其中 logits[l1:r1+1] 为某个triple的所有(正反向)candidates对应的输出
    #   golds[l1:r1+1] 为该triple所有candidates的label, 正常而言应该有且只有一个1, 其他是0
    # v_set = [[ques, m1, m2, rel, tag], ...], 即完整测试集; 仅在output_full_result时使用
    #   若 v_set==None, 需确保不用输出完整数据, 则过程中不会用到它
    # output_full_result, 是否输出所有triple的topk relation得分及tag, 用以debug

    if ranges==None:
        assert(False)
        # first time to run the method
        fn_evals = []
        v_set = []
        for i in range(1,8):
            fn_evals.append('./data/re_data/Dev_re_{}.dat'.format(i))
        for ix,fn_eval in enumerate(fn_evals):
            with open(fn_eval, 'r', encoding='utf-8') as fin:
                for line in fin:
                    v_set.append(line.strip().split('\t'))
        #v_set = v_set[:10000]
        ranges = []
        p1 = -1
        pre_q = None
        for p,cur_info in enumerate(v_set):
            cur_q = set()
            cur_q.update([cur_info[0],cur_info[1],cur_info[2]])
            if pre_q!=cur_q:
                pre_q = cur_q.copy()
                ranges.append([p1,p-1])
                p1 = p

        ranges = ranges[1:] + [[p1,p]]
    
    res_output_file = '../result/RE_topk_full/RE_top{}_'.format(topk) + time.ctime()[4:19].replace(':','_').replace(' ','_') + '.dat'
    assert(len(logits)==len(golds) and (v_set==None or len(golds)==len(v_set)))
    mrr, top_k_recall = 0,0
    for p1, p2 in ranges:
        output_str = ""
        # 将pred_score及gold_tag zip在一起, 并按得分排序
        c_true = golds[p1:p2+1]
        c_prob = logits[p1:p2+1]
        if v_set:
            c_zip = list(zip(c_prob, c_true, v_set[p1:p2+1]))
        else:
            c_zip = list(zip(c_prob, c_true))
        c_zip.sort(reverse=True, key=lambda x:x[0])
        cur_k = min(topk, p2-p1+1)
        
        # 计算gold得分排第几
        for i, itm in enumerate(c_zip):
            if itm[1]==1:
                rank = i
            if i < topk and output_full_result:   # 输出topk rel, 目前输出五元组及得分, 以\t连接
                output_str += itm[2][2]+'\t'+itm[2][0]+'\t'+itm[2][1]+'\t'+itm[2][3]+'\t'+str(itm[0])+'\t'+itm[2][4]+'\n'

        # 计算对应的topk_recall和MRR
        top_k_recall += (rank < topk)
        mrr += 1 / (rank + 1)
        if output_full_result:
            with open(res_output_file, 'a') as fout:
                fout.write(output_str+'\n')
        
    mrr /= len(ranges)
    top_k_recall /= len(ranges)
    print("MRR: {}. Top {} recall: {}.".format(mrr, topk, top_k_recall))
    return mrr, top_k_recall


def eval_re(model, dev_loader, log_output_file, output_model_file,
            epoch, best_mrr, topk, ranges, v_set, save_res=True, output_full_result=False):
    # evaluate MRR & top-k recall for RE task

    if hasattr(model, 'module'):
        model.module.eval()
    else:
        model.eval()
    golds, all_logits = [], []
    print("Num batches in dev_loader: ", len(dev_loader))
    for i, data in enumerate(dev_loader):
        if i%20==0:
            print("\rEvaluating... %s%%" % (100*(i+1)/len(dev_loader)), end='', flush=True)
        input_ids, type_masks, att_masks, tags = data
        input_ids, type_masks, att_masks = input_ids.cuda(), type_masks.cuda(), att_masks.cuda()
        with torch.no_grad():
            # 由于logits第0维是batch, 即便是dataparallel经过stack后也和正常一样, 不需要像loss一样mean
            logits = model(input_ids,\
                            enc_attention_mask=att_masks,\
                            token_type_ids=type_masks)
            golds += tags.tolist()
            all_logits += torch.nn.functional.softmax(logits, dim=1)[:,1].tolist()
    
    cur_mrr, topk_recall = show_mrr_and_topk_recall(all_logits, golds, topk, ranges, v_set, output_full_result)
    with open(log_output_file, 'a') as f_out:
        f_out.write("MRR: {}. Top {} recall: {}.\n".format(cur_mrr, topk, topk_recall))
        f_out.write("Current best MRR: {}.\n".format(max(cur_mrr, best_mrr)))
        print("Current best MRR: {}.\n".format(max(cur_mrr, best_mrr)))

    if cur_mrr > best_mrr:
        best_mrr = cur_mrr
        if save_res:
            print("Saving model...")
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), output_model_file)
            else:
                torch.save(model.state_dict(), output_model_file)

    if hasattr(model, 'module'):
        model.module.train()
    else:
        model.train()
    return best_mrr
