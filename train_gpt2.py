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
        self.c_proj.NANOGPT_SCALE_INIT=1 # self.c_proj设置一个标志属性 NANOGPT_SCALE_INIT，表示它将在后续初始化中使用 NanoGPT 定义的缩放初始化策略。
        
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
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True) # 加入flash attention
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
        self.c_proj.NANOGPT_SCALE_INIT=1

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

        # token嵌入层的权重与输出投影线性层权重绑定，GPT2的trick
        self.transformer.wte.weight=self.lm_head.weight

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear): # 若是线性层
            std=0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std*=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None):
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
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1)) # 把前2维合并起来(B,T)->(B*T)
        return logits,loss 
    
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
import tiktoken

class DataLoaderLite:
    def __init__(self,B,T):
        self.B=B
        self.T=T

        # 从硬盘加载数据，存到内存中
        with open('input.txt','r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position=0

    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position: self.current_position+B*T+1]
        x=(buf[:-1]).view(B,T) # 输入
        y=(buf[1:]).view(B,T) # targets
        # 前移current_position位置，为下一次取token准备
        self.current_position+=B*T
        # 如果下一次取样超出数据集，复位起始索引
        if self.current_position+(B*T+1)>len(self.tokens):
            self.current_position=0
        return x,y
    
# ----------------------------------------------------------------------------
# 生成前自动检测设备
import time

device='cpu'
if torch.cuda.is_available():
    device="cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device="mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader=DataLoaderLite(B=8,T=512)

torch.set_float32_matmul_precision('high') # 设置硬件float32计算的性能与精度水平，'high'设置为允许使用 TensorFloat-32（TF32），取性能和精度之间的一个平衡点

# 计算由x预测出的logits
model=GPT(GPTConfig(vocab_size=50304)) # 50304= 128*393，50304可以被更高的2的幂次数整除，更符合GPU的架构
model.to(device)
model=torch.compile(model) # 对模型编译，加速训练和推理，需torch2以上版本

# 优化！梯度下降
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4, betas=(0.9,0.95), eps=1e-8)
for i in range(50):
    t0=time.time()
    x,y=train_loader.next_batch()
    x,y=x.to(device),y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device,dtype=torch.bfloat16): # 利用自动混合精度的上下文管理器autocast，将model中部分运算操作的精度转换为bfloat16，可以提升效率并减少内存消耗
        logits,loss=model(x,y)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # 如果梯度范数大于1.0，则把所有参数的梯度按比例缩小，使范数变为1.0，这里返回的是裁剪前的梯度范数
    optimizer.step()
    torch.cuda.synchronize() # 等待GPU完成当前工作,确保time.time()计时的是GPU运行的时间
    t1=time.time()
    dt=t1-t0 # 以秒为单位展示耗时
    tokens_processed=train_loader.B*train_loader.T
    tokens_per_sec=tokens_processed/dt
    print(f"step {i:4d} | loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}") # 输出更漂亮些

import sys; sys.exit(0)

# 设置 prefix tokens
model.eval()
num_return_sequences=5
max_length=30
tokens=enc.encode("Hello, I'm a language model,")
tokens=torch.tensor(tokens,dtype=torch.long) # (8,)
tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1) # (5,8)在第0维加一个维度，并在第0维重复num_return_sequences次
x=tokens.to(device)

# 下面开始生成，x的形状为(B,T) 这里B=5，T=8
# 设置随机种子维42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1)<max_length:
    # 前向传播模型获取logits词表的输出结果
    with torch.no_grad():
        logits=model(x) # (B,T,vocab_size)
        # 截取logits序列长度维最后一个要素的vocab编号
        logits=logits[:,-1,:] # (B,vocab_size)
        # 计算输出词在各个词汇表中的概率分布
        probs=F.softmax(logits,dim=-1)
        # 取默认前50名的vocab（huggingface默认操作）
        # 下面得到topk_probs shape(5,50),topk_indices shape(5,50)
        topk_probs,topk_indices=torch.topk(probs,50,dim=-1)
        # 从中选取一个token
        # 多项式不要求输入向量的和为1
        ix=torch.multinomial(topk_probs,1) # (B,1)
        # 获取topk_indices第-1维的第ix个数据，也就是选取的token向量的索引
        xcol=torch.gather(topk_indices,-1,ix) # (B,1)
        # 放置到序列尾部
        x=torch.cat((x,xcol),dim=1)

# 打印生成的文本
for i in range(num_return_sequences):
    tokens=x[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)
