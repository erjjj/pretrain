'''
FineWeb-Edu 数据集
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
下载和tokenize数据，将数据分片保存到本地硬盘中
通过如下方式运行
$ python fineweb.py
保存tokenize后的数据分片到本地目录"edu_fineweb10B"
'''

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def tokenize(doc):
    # init the tokenizer
    enc=tiktoken.get_encoding('gpt2')
    eot=enc._special_tokens['<|endoftext|>'] # text token末尾符
    # tokenize一篇单独文本，返回一个uint16 token数组
    tokens=[eot] # 特殊字符<|endoftext|> token分割所有文档
    tokens.extend(enc.encode_ordinary(doc['text']))
    tokens_np=np.array(tokens)
    assert(0<=tokens_np).all() and (tokens_np<2**16).all(),"token dictionary too large for uint16"
    tokens_np_uint16=tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename,tokens_np):
    np.save(filename,tokens_np)


def main():
    # ----------------------------------------------------------------------------
    local_dir='edu_fineweb10B'
    remote_name='sample-10BT'
    shard_size=int(1e8) # 每个分片100Mtokens，一共100个shards

    # 创建本地目录缓存
    DATA_CACHE_DIR=os.path.join(os.path.dirname(__file__),local_dir)
    os.makedirs(DATA_CACHE_DIR,exist_ok=True)

    # 下载数据集
    fw=load_dataset("HuggingFaceFW/fineweb-edu",name=remote_name,split='train')



    # tokenize所有文本
    nprocs=max(1,os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index=0
        # 预分配缓存，存储当前切片
        all_tokens_np=np.empty((shard_size,),dtype=np.uint16)
        token_count=0
        progress_bar=None
        for tokens in pool.imap(tokenize,fw,chunksize=16):

            # 新tokens在当前shard是否还有充足的空间
            if token_count+len(tokens)<shard_size:
                # 在当前shard后加tokens即可
                all_tokens_np[token_count:token_count+len(tokens)]=tokens
                token_count+=len(tokens)
                # 更新进度条
                if progress_bar is None:
                    progress_bar=tqdm(total=shard_size,unit='tokens',desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # 把当前shard写进内存，再开一个新shard
                split='val' if shard_index==0 else 'train'
                filename=os.path.join(DATA_CACHE_DIR,f"edufineweb_{split}_{shard_index:06d}")
                # 
                remainder=shard_size-token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder]=tokens[:remainder]
                write_datafile(filename,all_tokens_np)
                shard_index+=1
                progress_bar=None
                # 更新下一个切片，且切片以当前token的剩余部分为开头
                all_tokens_np[0:len(tokens)-remainder]=tokens[remainder:]
                token_count=len(tokens)-remainder

        # 把剩下的tokens存到最后一个切片
        if token_count!=0:
            split='val' if shard_index==0 else 'train'
            filename=os.path.join(DATA_CACHE_DIR,f'edufineweb_{split}_{shard_index:06d}')
            write_datafile(filename,all_tokens_np[:token_count])


if __name__=='__main__':
    mp.freeze_support()
    main()