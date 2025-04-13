import os
import math
import time
import inspect
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

    def configure_optimizers(self,weight_decay,learning_rate,device):
        # 从所有需要梯度下降的参数开始筛选
        param_dict={pn:p for pn,p in self.named_parameters()}
        param_dict={pn:p for pn,p in param_dict.items() if p.requires_grad}
        # 创建optim groups，只有2D参数才会权重衰减
        # 即所有参与矩阵乘法和嵌入的参数会衰减，所有bias参数和layernorms层的参数不衰减
        decay_params=[p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params=[p for n,p in param_dict.items() if p.dim()<2]
        optim_groups=[
            {'params':decay_params,'weight_decay':weight_decay},
            {'params':nodecay_params,'weight_decay':0.0}
        ]
        num_decay_parmas=sum(p.numel() for p in decay_params)
        num_nodecay_params=sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_parmas:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 构建AdamW优化器，并采用fused模式，加速训练
        fused_available='fused' in inspect.signature (torch.optim.AdamW).parameters
        use_fused=fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer=torch.optim.AdamW(optim_groups,lr=learning_rate,betas=(0.9,0.95),eps=1e-8,fused=use_fused)
        return optimizer
    
# ----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt=np.load(filename)
    ptt=torch.tensor(npt,dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_processes,split):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes=num_processes
        assert split in {'train','val'}

        # 获取切片文件名
        data_root='edu_fineweb10B'
        shards=os.listdir(data_root)
        shards=[s for s in shards if split in s]
        shards=sorted(shards)
        shards=[os.path.join(data_root,s) for s in shards]
        self.shards=shards
        assert len(shards)>0,f'no shards found for split {split}'
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, 从第1个shard开始初始化
        self.current_shard=0
        self.tokens=load_tokens(self.shards[self.current_shard])
        self.current_position=self.B*self.T*self.process_rank
    
    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position: self.current_position+B*T+1]
        x=(buf[:-1]).view(B,T) # 输入
        y=(buf[1:]).view(B,T) # targets
        # 前移current_position位置，为下一次取token准备
        self.current_position+=B*T*self.num_processes
        # 如果下一次取样超出数据集，来到下一个切片
        if self.current_position+(B*T*self.num_processes+1)>len(self.tokens):
            self.current_shard=(self.current_shard+1)%len(self.shards)
            self.tokens=load_tokens(self.shards[self.current_shard])
            self.current_position=B*T*self.process_rank
        return x,y
    
# ----------------------------------------------------------------------------
# 简单运行
# python train_gpt2.py
# 分布式数据并行运行启动方式，以2个GPU运算（Karpathy是8个GPU，我这里有两个GPU）
# torchrun --standalone --nproc_per_node=2 train_gpt2.py

# 运行循环训练
from torch.distributed import init_process_group,destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 设置DDP，分布式数据并行(distributed data parallel)
# torchrun命令要设置环境变量，RANK,LOCAL_RANK,WORLD_SIZE
ddp=int(os.environ.get('RANK',-1))!=-1 # 判断此次运行是否是ddp
if ddp:
    # ddp运行需要CUDA，我们需要根据进程的标识符设置合适的设备
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank=int(os.environ['RANK']) # 当前进程的全局编号，每个训练进程的唯一ID
    ddp_local_rank=int(os.environ['LOCAL_RANK']) # 当前进程的GPU编号
    ddp_world_size=int(os.environ['WORLD_SIZE']) # DDP模式下并行的进程总数
    device=f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process=ddp_rank==0 # 判断是否是主进程，负责保存模型、打印日志、验证模型
else:
    # 非DDP的普通运行方式
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1
    master_process=True
    # 生成前自动检测设备
    device='cpu'
    if torch.cuda.is_available():
        device="cuda"
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device="mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc=tiktoken.get_encoding('gpt2')

total_batch_size=524288 # 2^19, ~0.5M, 数量的tokens
B=8 # 小批量batch
T=512 # 序列长度
assert total_batch_size%(B*T*ddp_world_size)==0, "make sure total_batch_size is divisible by B*T*ddp_world_size"
grad_accum_steps=total_batch_size//(B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader=DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='train')
val_loader=DataLoaderLite(B=B,T=T,process_rank=ddp_rank,num_processes=ddp_world_size,split='val')

torch.set_float32_matmul_precision('high') # 设置硬件float32计算的性能与精度水平，'high'设置为允许使用 TensorFloat-32（TF32），取性能和精度之间的一个平衡点

# 创建模型
model=GPT(GPTConfig(vocab_size=50304)) # 50304= 128*393，50304可以被更高的2的幂次数整除，更符合GPU的架构
model.to(device)
model=torch.compile(model) # 对模型编译，加速训练和推理，需torch2以上版本
if ddp:
    model=DDP(model,device_ids=[ddp_local_rank])
raw_model=model.module if ddp else model # 原始未封装的模型

max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=715
max_steps=19073
def get_lr(it):
    # 1) 对于前warmup_iters步steps，线性提高学习率
    if it<warmup_steps:
        return max_lr*(it+1)/warmup_steps
    # 2) 若超出了学习率衰减阶段，采用最低学习率
    if it>max_steps:
        return min_lr
    # 3) warmup_steps与max_steps之间，采用cos衰减到最小学习率
    decay_ratio=(it-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio)) # decay_ratio从0升到1，cos从1降为-1，coeff从1降为0
    return min_lr+coeff*(max_lr-min_lr)

# 优化！梯度下降
optimizer=raw_model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device=device)

for step in range(max_steps):
    t0=time.time()

    # 每隔指定步数评估一下验证集的损失
    if step%100==0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum=0.0
            val_loss_steps=20
            for _ in range(val_loss_steps):
                x,y=val_loader.next_batch()
                x,y=x.to(device),y.to(device)
                with torch.autocast(device_type=device,dtype=torch.bfloat16):
                    logits,loss=model(x,y)
                loss=loss/val_loss_steps
                val_loss_accum+=loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum,op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
    
    # 一旦从模型中开始生成时
    # torch.compile会报错，所以下面这段代码先用 and False失效掉
    # 停止torch.compile(),可以正常生成
    if step>0 and step%100==0:
        model.eval()
        num_return_sequences=4
        max_length=32
        tokens=enc.encode("Hello, I'm a language model,")
        tokens=torch.tensor(tokens,dtype=torch.long) # (8,)
        tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1) # (5,8)在第0维加一个维度，并在第0维重复num_return_sequences次
        xgen=tokens.to(device)
        sample_rng=torch.Generator(device=device)
        sample_rng.manual_seed(42+ddp_rank)
        while xgen.size(1)<max_length:
            # 前向传播模型获取logits词表的输出结果
            with torch.no_grad():
                logits,loss=model(xgen) # (B,T,vocab_size)
                # 截取logits序列长度维最后一个要素的vocab编号
                logits=logits[:,-1,:] # (B,vocab_size)
                # 计算输出词在各个词汇表中的概率分布
                probs=F.softmax(logits,dim=-1)
                # 取默认前50名的vocab（huggingface默认操作）
                # 下面得到topk_probs shape(5,50),topk_indices shape(5,50)
                topk_probs,topk_indices=torch.topk(probs,50,dim=-1)
                # 从中选取一个token
                # 多项式不要求输入向量的和为1
                ix=torch.multinomial(topk_probs,1,generator=sample_rng) # (B,1)
                # 获取topk_indices第-1维的第ix个数据，也就是选取的token向量的索引
                xcol=torch.gather(topk_indices,-1,ix) # (B,1)
                # 放置到序列尾部
                xgen=torch.cat((x,xcol),dim=1)
        # 打印生成的文本
        for i in range(num_return_sequences):
            tokens=xgen[i,:max_length].tolist()
            decoded=enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # 开启训练loop
    model.train()
    optimizer.zero_grad()
    loss_accum=0.0
    for micro_step in range(grad_accum_steps): 
        x,y=train_loader.next_batch()
        x,y=x.to(device),y.to(device)
        with torch.autocast(device_type=device,dtype=torch.bfloat16): # 利用自动混合精度的上下文管理器autocast，将model中部分运算操作的精度转换为bfloat16，可以提升效率并减少内存消耗
            logits,loss=model(x,y)
        # 在grad_accum_steps内，要放缩loss，以适应梯度累积
        # backward()会累积各个grad_accum_steps下的梯度
        # 得到的是grad_accum_steps倍的参数矩阵的梯度
        # 我们要的是梯度均值，除以梯度累计的次数即可
        loss=loss/grad_accum_steps
        loss_accum+=loss.detach()
        if ddp:
            model.require_backward_grad_sync=(micro_step==grad_accum_steps-1)
        loss.backward() # 求的是optimizer.zero_grad()前的梯度累积，而不是这里loss的梯度
    if ddp:
        dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) # 如果梯度范数大于1.0，则把所有参数的梯度按比例缩小，使范数变为1.0，这里返回的是裁剪前的梯度范数
    # 设置此轮epoch的学习率
    lr=get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr']=lr
    optimizer.step()
    torch.cuda.synchronize() # 等待GPU完成当前工作,确保time.time()计时的是GPU运行的时间
    t1=time.time()
    dt=t1-t0 # 以秒为单位展示耗时
    tokens_processed=train_loader.B*train_loader.T*grad_accum_steps*ddp_world_size
    tokens_per_sec=tokens_processed/dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}") # 输出更漂亮些

if ddp:
    destroy_process_group()