import torch 
import torch.nn as nn
import torch.nn.functional as F  
import math

def scaled_dot_product(q,k,v,mask=None):
    #q : 30 x 8 x 200 k : 30 x 8 x 200 v : 30 x 8 x 200  mask : 200 x 200
    d_k=q.size(-1)
    scaled=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(d_k)#d-k is use for scale # 30 x8 x 200 x 200
    if mask is not None:
        scaled+=mask#30 x 8 x 200 x 200
    attention=F.softmax(scaled,dim=-1)# 30 x 8 x 200 x 200
    values=torch.matmul(attention,v)# 30 x 8 x 200 x 64
    return values,attention

class LayerNormalization(nn.Module):
    def __init__(self,parameters_shape,eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
    
    def forward(self, inputs):
        dims=[-(i+1) for i in range(len(self.parameters_shape))] # it takes last layer
        mean=inputs.mean(dim=dims,keepdim=True)
        var=((inputs-mean)**2).mean(dim=dims,keepdim=True)
        std=(var+self.eps).sqrt()
        y=(inputs-mean)/std
        out=self.gamma * y + self.beta
        return out
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model,hidden,drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        self.qkv_layer=nn.Linear(d_model,3*d_model)# it means map d_model to 3*d_model which means 512 to 1536
        self.linear_layer=nn.Linear(d_model,d_model)

    def forward(self,x,mask=None):
        batch_size,sequence_length,d_model=x.size()#it is a target sentence 30 x 200 x512
        qkv=self.qkv_layer(x)# 30 x 200 x 1536
        qkv=qkv.reshape(batch_size,sequence_length,self.num_heads,3*self.head_dim)# we reshaping because we want to perform multihead attention
        # 300 x 200 x  8 x 192
        qkv=qkv.permute(0,2,1,3)#30 x 8 x 200 x 192
        q,k,v=qkv.chunk(3,dim=-1)#30 x8 x 200 x 64  for single q k and v
        values,attention=scaled_dot_product(q,k,v,mask)
        values=values.reshape(batch_size,sequence_length,self.num_heads*self.head_dim)# 30 x 200 x 512 # concatenating together 8 heads
        out =self.linear_layer(values)# 30 x 200 x 512 # 
        return out
class MultiheadCrossAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
        self.kv_layer=nn.Linear(d_model,2*d_model)
        self.q_layer=nn.Linear(d_model,d_model)
        self.linear_layer=nn.Linear(d_model,d_model)
    def forward(self,x,y,mask=None):
        batch_size,sequence_length,d_model=x.size()
        kv=self.kv_layer(x)# 30 x 200 x 1024
        q=self.q_layer(y)
        kv=kv.reshape(batch_size,sequence_length,self.num_heads,2*self.head_dim)#30 x 200 x 8 x 128
        q=q.reshape(batch_size,sequence_length,self.num_heads,self.head_dim)# 30 x 200 x 8 x 64
        kv=kv.permute(0,2,1,3)#30 x 8 x 200 x 128
        q=q.permute(0,2,1,3)#30 x 8 x 200 x 64
        k,v=kv.chunk(2,dim=-1)
        values,attention=scaled_dot_product(q,k,v,mask)
        values=values.reshape(batch_size,sequence_length,d_model)
        out=self.linear_layer(values)

        return out
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        super().__init__()
        self.self_attention=MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1=LayerNormalization(parameters_shape=[d_model])
        self.dropout1=nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention=MultiheadCrossAttention(d_model=d_model,num_heads=num_heads)
        self.norm2=LayerNormalization(parameters_shape=[d_model])
        self.dropout2=nn.Dropout(p=drop_prob)
        self.ffn=PositionWiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm3=LayerNormalization(parameters_shape=[d_model])
        self.dropout3=nn.Dropout(p=drop_prob)
    def forward(self,x,y,decoder_mask):
        _y=y # Residual connection
        y=self.self_attention(y,mask=decoder_mask)
        y=self.dropout1(y)
        y=self.norm1(y+_y)
        _y=y# Another Residual
        y=self.encoder_decoder_attention(y,x,mask=None)
        y=self.dropout2(y)
        y=self.norm2(y+_y)
        _y=y 
        y=self.ffn(y)
        y=self.dropout3(y)
        y=self.norm3(y+_y)
        return y

class SequentialDecoder(nn.Sequential):# we using sequential decoder because we have more than one parameter in Decoder forward .
        def forward(self, *inputs):
            x,y,mask = inputs
            for module in self._modules.values():
                y=module(x,y,mask)# our same value of x is feeded in decoder layer but y is calculated which produce new word or tokens and it is again feeded to decoder.
            return y

class Decoder(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers=1):
        super().__init__()
        self.layers =SequentialDecoder(*[DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob) for _ in range(num_layers)])
        # nn.sequential is not used here because forward contain more than one parameter

    def forward(self,x,y,mask):#input sentences,target sentence,mask
        # x :30 x 200 x512
        # y:30 x 200 x512
        # Mask : 200 x 200
        y=self.layers(x,y,mask)
        return y
    
if __name__ == "__main__":
    d_model=512
    num_heads=8
    drop_prob=0.1
    batch_size=30
    max_sequence_length=200
    ffn_hidden=2048
    num_layers=5
    x=torch.randn((batch_size,max_sequence_length,d_model))# input lang positional encoded
    y=torch.randn((batch_size,max_sequence_length,d_model))# target lang posiitonal encoded
    mask=torch.full([max_sequence_length,max_sequence_length],float('-inf'))
    mask=torch.triu(mask,diagonal=1)
    decoder=Decoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers)
    out=decoder(x,y,mask)
    print(out.size())
