import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# ----------------------------------------------------------------------------
# 自注意力
class CausalSelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        # 确保嵌入维度被头数整除
        assert config.n_embd % config.n_head ==0
        # 一个batch中q，k，v在各个head（所有head）的计算
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        # head合并后用于整合的线性变换层
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        
        self.n_head=config.n_head
        self.n_embd=config.n_embd
        # 创建一个mask,生成一个下三角矩阵，矩阵的主对角线及以下元素为1，其余为0
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
        B,T,C = x.size() # batch size, sequence length, embedding dimensionality(n_embd)
        # 计算一个batch中所有head的q,k,v，移动head维的维度
        # nh为head数量，hs为单个head的维度，有C（channel数）=nh*hs
        # 以GPT(124M)为例，nh=12，hs=64，通道数C=nh*hs=768
        qkv=self.c_attn(x)
        q,k,v=qkv.split(self.n_embd,dim=2)
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # 分割-维度交换(B,nh,T,hs)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) 
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        # 注意力计算，对每个head（每对q,k)计算注意力方阵(T,T)
        att=(q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf')) # 保留att下三角矩阵的值，其余部分设置为0，让att中的相应行只能看到这一行之前的内容
        att=F.softmax(att,dim=-1) # 最后一维softmax，将方形矩阵的每一行转为概率权重
        y=att@v # (B,nh,T,T)@(B,nh,T,hs)->(B,nh,T,hs)
        y=y.transpose(1,2).contiguous().view(B,T,C) # 逐头重组得
        # 自注意力层输出
        y=self.c_proj(y)
        return y
    
class MLP(nn.Module):
    # 全连接层
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x
    
class Block(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x)) # 先归一化层-注意力层-残差链接
        x=x+self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size:int=1024 # 读取最大序列长度
    vocab_size:int=50257 # token数量，50000次BPE生成+256基础字符+1个<|endoftext|>token
    n_layer:int=12 # 层数
    n_head:int=12 # 头数
    n_embd:int=768 # 嵌入维度

class GPT(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)

    def forward(self,idx):
        # 作为输入，idx(B,T)(Batch_size,序列长度)
        B,T=idx.size()
        assert T<=self.config.block_size,f"cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # foward前向传播token和位置编码
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device) # shape:(T)
        pos_emb=self.transformer.wpe(pos) # 位置嵌入，shape(T,n_embd)
        tok_emb=self.transformer.wte(idx) # token嵌入，shape(B,T,n_embd)
        x=tok_emb+pos_emb
        # 前向传播transformer模块
        for block in self.transformer.h:
            x=block(x)
        # 前向传播最后的layernorm和classifier
        x=self.transformer.ln_f(x)
        logits=self.lm_head(x) # shape(B,T,vocab_size)
        return logits 
    
    @classmethod
    def from_pretrained(cls,model_type):
        '''从huggingface加载预训练好的GPT-2模型'''
        assert model_type in {'gpt2','gpt2-medium','gpt2-large','gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # 层数，头数，嵌入维度取决于下面定义的model_type
        config_args={
            'gpt2':         dict(n_layer=12,n_head=12,n_embd=768),  # 124M 参数
            'gpt2-medium':  dict(n_layer=24,n_head=16,n_embd=1024), # 350M 参数
            'gpt2-large':   dict(n_layer=36,n_head=20,n_embd=1280), # 774M 参数
            'gpt2-xl':      dict(n_layer=48,n_head=25,n_embd=1600), # 1558M 参数
        }[model_type]
        config_args['vocab_size']=50257 #
        config_args['block_size']=1024 #
        # 下面创建一个从0开始的初始化minGPT模型
        config=GPTConfig(**config_args)
        model=GPT(config)
        sd=model.state_dict()
        sd_keys=sd.keys()
        sd_keys=[k for k in sd_keys if not k.endswith('.attn.bias')] # 去除掩膜，生成时不需要盖住

        # 初始化一个huggingface/Transformers 模型
        model_hf=GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf=model_hf.state_dict()

        # 确保所有参数与我们设计的对齐，形状匹配
        sd_keys_hf=sd_hf.keys()
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略buffer
        sd_keys_hf=[k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        # openai使用了Conv1D模块，我们只需要一个简单的线性层
        # 这意味着我们引用GPT2的权重到我们定义的模型矩阵时，需要转置一部分
        transposed=['attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight']
        assert len(sd_keys_hf)==len(sd_keys),f"mismatched keys: {len(sd_keys_hf)}!={len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 转置Conv1D部分的权重
                assert sd_hf[k].shape[::-1]==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 其它部分引入时简单地copy即可
                assert sd_hf[k].shape==sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
# ----------------------------------------------------------------------------
model=GPT.from_pretrained('gpt2')
print("didn't crash yay!")