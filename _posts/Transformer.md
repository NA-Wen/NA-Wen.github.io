---
title: Transformer
date: 2023-08-20 11:13:11
index_img: /cover/1.jpg
author : NA-Wen
tags: [NLP,transformer]
category: NLPBasic 
---
# Transformer

seq2seq

RNN: it is hard to parallel

CNN : only higher level can get the seq information

self-attention : 

parallel and  can see the whole sentence
<!--more-->
 

## The whole process:

Encoder : 

1. get the vector of each word (word feature + word position): embedding 
2. get the words matrix , through several (6) encoder block , and get the same dim matrix

Decoder:

use the matrix and a masked word map 

have 1~i(i+1 and after i+1 masked) ,and translate the i+1 word 

### embedding

单词的 Embedding 有很多种方式可以获取，例如可以采用 Word2Vec、Glove 等算法预训练得到，也可以在 Transformer 中训练得到。

Transformer 中使用位置 Embedding 保存单词在序列中的相对或绝对位置。

PE : the same dim with word embedding ; we can get this from calculating.

![v2-8b442ffd03ea0f103e9acc37a1db910a_1440w.png](/img/v2-8b442ffd03ea0f103e9acc37a1db910a_1440w.png)

d is the dimension of word embedding (in another word ,  the PE embedding) 

advantages: 

1. can suit sentences longer than d 
2. can easily calculate the relative position : we can get PE(pos+k) from PE(pos)

## Self-attention

![v2-f6380627207ff4d1e72addfafeaff0bb_1440w.webp](/img/v2-f6380627207ff4d1e72addfafeaff0bb_1440w.webp)

Multi-Head Attention contains several self-attention ;

一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。

### single self-attention

![v2-6444601b4c41d99e70569b0ea388c3bd_1440w.png](/img/v2-6444601b4c41d99e70569b0ea388c3bd_1440w.png)

a=Wx  embedding 

a→ self attention  layer(matrix) : 3 q k v

q :query(match others) k:  key(to be matched)  v: information to be extracted 

each query to do attention for each key(attention : get two vec and get a score , representing similarity)

Attention : 

$$
Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_k}})V
$$

### multi self-attention

 we use some self-attention and get some results for each ; then we concat all results , pass a linear layer ,then get the final output  matrix   

> the result of single self-attention is the same shape with Q or K  or V ;the result of multi self-attention is the same shape with input word matrix due to the final linear layer
> 

## Encoder

Add & Norm : 

![v2-a4b35db50f882522ee52f61ddd411a5a_1440w.png](/img/v2-a4b35db50f882522ee52f61ddd411a5a_1440w.png)

Feedforward : 

Feed Forward 层比较简单，是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。

**X**是输入，Feed Forward 最终得到的输出矩阵的维度与**X**一致.

<aside>
💡 Through the above process , first do embedding , then use the Multi-Head Attention, Feed Forward, Add & Norm as a encoder block , several encoder blocks form an encoder. The final output is the input of the decoder

</aside>

## Deocder

1. masked multi-head attention
    
    masked：翻译是根据顺序进行翻译的；在翻译当前单词时只能获取到前序单词的信息。
    
    so we do dot product with a masked matrix before softmax  ;then product with V and get the Z
    
    ![0820.png](/img/0820.png)
    
    teacher forcing？
    
2. multi head attention
    
    K V if from the encoder result , not from the former result.
    
    the advantage is that when doing decoder , under each step ,all information in encoder can be used.
    
3. output 
    
    we use a softmax in the final step ; and according to the masked multi-head attention , the i-th row in the matrix Z contains all words before i+1. So we predict the i-th word from the i-1th row.   
    

## understanding

1. attention
    
    > An attention function can be described as a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, **where the weight assigned to each value is computed by a compatibility of the query with the corresponding key.**
    > 

We use scaled-dot product here.

1. padding mask
    
    Due to the different length of each sequences , we append -xxxx after the short sequences ; so after softmax , the probability will come to 0 
    
    ## code
    
    ```python
    class EncoderLayer(nn.Module):
    	"""Encoder的一层。"""
    
        def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
            super(EncoderLayer, self).__init__()
    
            self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
            self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
    
        def forward(self, inputs, attn_mask=None):
    
            # self attention
            context, attention = self.attention(inputs, inputs, inputs, padding_mask)
    
            # feed forward network
            output = self.feed_forward(context)
    
            return output, attention
    
    class Encoder(nn.Module):
    	"""多层EncoderLayer组成Encoder。"""
    
        def __init__(self,
                   vocab_size,
                   max_seq_len,
                   num_layers=6,
                   model_dim=512,
                   num_heads=8,
                   ffn_dim=2048,
                   dropout=0.0):
            super(Encoder, self).__init__()
    
            self.encoder_layers = nn.ModuleList(
              [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
               range(num_layers)])
    
            self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
            self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
    
        def forward(self, inputs, inputs_len):
            output = self.seq_embedding(inputs)
            output += self.pos_embedding(inputs_len)
    
            self_attention_mask = padding_mask(inputs, inputs)
    
            attentions = []
            for encoder in self.encoder_layers:
                output, attention = encoder(output, self_attention_mask)
                attentions.append(attention)
    
            return output, attentions
    class DecoderLayer(nn.Module):
    
        def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
            super(DecoderLayer, self).__init__()
    
            self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
            self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
    
        def forward(self,
                  dec_inputs,
                  enc_outputs,
                  self_attn_mask=None,
                  context_attn_mask=None):
            # self attention, all inputs are decoder inputs
            dec_output, self_attention = self.attention(
              dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
    
            # context attention
            # query is decoder's outputs, key and value are encoder's inputs
            dec_output, context_attention = self.attention(
              enc_outputs, enc_outputs, dec_output, context_attn_mask)
    
            # decoder's output, or context
            dec_output = self.feed_forward(dec_output)
    
            return dec_output, self_attention, context_attention
    
    class Decoder(nn.Module):
    
        def __init__(self,
                   vocab_size,
                   max_seq_len,
                   num_layers=6,
                   model_dim=512,
                   num_heads=8,
                   ffn_dim=2048,
                   dropout=0.0):
            super(Decoder, self).__init__()
    
            self.num_layers = num_layers
    
            self.decoder_layers = nn.ModuleList(
              [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
               range(num_layers)])
    
            self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
            self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
    
        def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
            output = self.seq_embedding(inputs)
            output += self.pos_embedding(inputs_len)
    
            self_attention_padding_mask = padding_mask(inputs, inputs)
            seq_mask = sequence_mask(inputs)
            self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)
    
            self_attentions = []
            context_attentions = []
            for decoder in self.decoder_layers:
                output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask)
                self_attentions.append(self_attn)
                context_attentions.append(context_attn)
    
            return output, self_attentions, context_attentions
    class Transformer(nn.Module):
    
        def __init__(self,
                   src_vocab_size,
                   src_max_len,
                   tgt_vocab_size,
                   tgt_max_len,
                   num_layers=6,
                   model_dim=512,
                   num_heads=8,
                   ffn_dim=2048,
                   dropout=0.2):
            super(Transformer, self).__init__()
    
            self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                                   num_heads, ffn_dim, dropout)
            self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                                   num_heads, ffn_dim, dropout)
    
            self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
            self.softmax = nn.Softmax(dim=2)
    
        def forward(self, src_seq, src_len, tgt_seq, tgt_len):
            context_attn_mask = padding_mask(tgt_seq, src_seq)
    
            output, enc_self_attn = self.encoder(src_seq, src_len)
    
            output, dec_self_attn, ctx_attn = self.decoder(
              tgt_seq, tgt_len, output, context_attn_mask)
    
            output = self.linear(output)
            output = self.softmax(output)
    
            return output, enc_self_attn, dec_self_attn, ctx_attn
    ```