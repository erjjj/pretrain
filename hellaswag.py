'''
下载并评估HellaSwag数据集
https://github.com/rowanz/hellaswag

HellaSwag json样本演示

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: 该样本的ActivityNet或WikiHow标签
context: 看不懂。There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: (0,1,2,3)
split: 数据集的用途：train,val,test
split_type:
source_id: 数据来源编号

gpt2 (124M)
 - eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
 - this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)
 
 gpt2-xl (1558M)
 - eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
 - this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

 HellaSwag中的验证集有10042个样本
'''

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
DATA_CACHE_DIR=os.path.join(os.path.dirname(__file__),'hellaswag')

def download_file(url: str,fname:str,chunk_size=1024):
    '辅助函数：给定url下载文件'
    resp=requests.get(url,stream=True)
    total=int(resp.headers.get('content-length',0))
    with open(fname,'wb') as file,tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size=file.write(data)
            bar.update(size)

hellaswags={
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
     "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
     "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc=tiktoken.get_encoding('gpt2')

def download(split):
    '下载HellaSwag数据到目录'
    os.makedirs(DATA_CACHE_DIR,exist_ok=True)
    data_url=hellaswags[split]
    data_filename=os.path.join(DATA_CACHE_DIR,f'hellaswag_{split}.jsonl')
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url,data_filename)

def render_example(example):
    '''
    example是一个字典，分别映射到3个torch tensors
    - tokens 上下文+多个候选答案的拼接
    - mask 候选答案的位置，只在这计算损失
    - label 表示正确候选答案的选项
    '''
    ctx=example['ctx']
    label=example['label']
    endings=example['endings']

    # 复现评估集数据
    data={
        'label':label,
        'ctx_tokens':None,
        'ending_tokens':[],
    }

    # 收集所有的tokens
    ctx_tokens=enc.encode(ctx)
    data['ctx_tokens']=ctx_tokens
    tok_rows=[]
    mask_rows=[]
    for end in endings:
        end_tokens=enc.encode(' '+end) # 前面加' '，以适应GPT2的tokenizer
        tok_rows.append(ctx_tokens+end_tokens)
        mask_rows.append([0]*len(ctx_tokens)+[1]*len(end_tokens))
        data['ending_tokens'].append(end_tokens)

    # 校对时注意每行的token数不一定相等
    max_len=max(len(row) for row in tok_rows)
    tokens=torch.zeros((4,max_len),dtype=torch.long)
    mask=torch.zeros((4,max_len),dtype=torch.long)
    for i,(tok_row,mask_row) in enumerate(zip(tok_rows,mask_rows)):
        tokens[i,:len(tok_row)]=torch.tensor(tok_row)
        mask[i,:len(mask_row)]=torch.tensor(mask_row)

    return data,tokens,mask,label

def iterate_examples(split):
    # 10042个例子
    download(split)
    with open(os.path.join(DATA_CACHE_DIR,f'hellaswag_{split}.jsonl'),'r') as f:
        for line in f:
            example=json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type,device):

    torch.set_float32_matmul_precision('high')
    model=GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model=torch.compile(model) # 视需要

    num_correct_norm=0
    num_correct=0
    num_total=0
    for example in iterate_examples('val'):
        data,tokens,mask,label=render_example(example)
        tokens=tokens.to(device)
        mask=mask.to(device)

        # 计算logits
        logits=model(tokens).logits
        # 计算所有位置的自回归损失
        shift_logits=(logits[...,:-1,:]).contiguous()
        shift_tokens=(tokens[...,1:]).contiguous()
        flat_shift_logits=shift_logits.view(-1,shift_logits.size(-1))
        flat_shift_tokens=shift_tokens.view(-1)
        shift_losses=F.cross_entropy(flat_shift_logits,flat_shift_tokens,reduction='none')
        shift_losses=shift_losses.view(tokens.size(0),-1)
        # 只计算每行的答案区域（mask==1）的平均损失
        shift_mask=(mask[...,1:]).contiguous() # 移动mask，从最后一个prompt token开始
        masked_shift_losses=shift_losses*shift_mask
        # mask label损失求和再平均
        sum_loss=masked_shift_losses.sum(dim=1)
        avg_loss=sum_loss/shift_mask.sum(dim=1)
        # 现在我们对每4个候选答案由一个损失
        # 损失值最低的答案就是我们的结果
        pred=sum_loss.argmin().item()
        pred_norm=avg_loss.argmin().item()

        # 累计结果
        num_total+=1
        num_correct+=int(pred==label)
        num_correct_norm+=int(pred_norm==label)
        print(f'{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}')

        # 打印一些结果判断例子，展示损失
        if num_total<10:
            print('---')
            print(f"Context:\n {example['ctx']}")
            print(f'Endings:')
            for i,end in enumerate(example['endings']):
                print(f'{i} (loss:{avg_loss[i].item():.4f}) {end}')
            print(f'predicted: {pred_norm}, actual: {label}')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--model_type',type=str,default='gpt2',help='the model type to use')
    parser.add_argument('-d','--device',type=str,default='cuda',help='the device to use')
    args=parser.parse_args()
    evaluate(args.model_type,args.device)