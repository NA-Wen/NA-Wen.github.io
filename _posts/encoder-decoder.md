---
title: why decoder only in LLM?
date: 2023-09-05 18:47:20
index_img: /cover/3.jpg
author : NA-Wen
category: NLP
tags: [NLP,decoder-only,architecture]
catagories: NLP
---
# why decoder-only in LLM ?
至今， Decoder-only的架构在NLU & NLG任务中占据着越来越重要的地位。相较于ED，LM的优势在于何处？
<!--more-->
# Intro

我们发现，在2022年OpenAI 推出ChatGPT之后，一直到2023年至今， Decoder-only的架构在NLU & NLG任务中占据着越来越重要的地位。相关的工作数量也逐渐超过Encoder-only 和Encoder-Decoder 架构的相关工作。

![](/img1/Untitled.png)

之前google在T5这篇工作中，详细对比了不同的模型架构（encoder-decoder, decoder/language model , prefix LM）并做了对比实验，结果如下。

![](/img1/Untitled%201.png)
可以看到，不共享参数的encoder-decoder表现优于其他的架构。工作中称，虽然不共享参数的encoder-decoder架构参数量是 2P ，但是其推理时间和参数量为 P 的decoder-only 架构是相近的。且不共享参数的Enc-dec表现上要略好于共享参数的Enc-dec。

T5的参数只在百亿左右，在目前GPT的千亿参数级别，decoder-only的架构相较Enc-dec更加占据优势（不过GLM130B和UPalm-540B不是decoder-only）。对于此模型架构上的趋势变化，苏神给出的回答是：

> 输入部分的注意力改为双向不会带来收益，Encoder-Decoder架构的优势很可能只是源于参数翻倍。
> 

也有说法称，zero-shot的情况下，decoder-only表现最好（对于LLM 来说，抛弃fine-tuning走向zero-shot是最重要的一个feature 🥰）。

还有说法称，因为对于decoder-only的scaling law被验证，因此OpenAI敢于直接scale up。而其他的模型架构上并没有类似的验证过程。所以现在scale up后，decoder-only的架构outperform了。

前两种说法，分别从理论证明和实验验证的不同角度出发。笔者查询了相关文献，记录了简要的笔记和个人的一些想法。

## 低秩问题

> Bhojanapalli, S., Yun, C., Rawat, A.S., Reddi, S. and Kumar, S., 2020, November. Low-rank bottleneck in multi-head attention models. In *International conference on machine learning* (pp. 864-873). PMLR.
> 

证明了attention会出现因为低秩而表达能力下降的原因，为之后的思考提供了基础。

首先，我们规定记号如下：

$$
W_k \in \R^{d_k \times d },W_q \in \R^{d_q \times d },W_v \in \R^{d_v \times d }\\X\in \R^{d\times n}
$$

X是输入矩阵，W是生成query，key，value的矩阵。n代表token数目，d代表token维度。

$$
Attention(X)=W_vX\;softmax(\frac{(W_kX)^T(W_qX)}{\sqrt{d_k}})\in \R ^{d_v\times n}\\P=softmax(\frac{(W_kX)^T(W_qX)}{\sqrt{d_k}})\in \R ^{n\times n}
$$

在经过attention层之后，还有一个带残差的全连接层+LN

$$
LN(X+W_o\:Attention(X))
$$

X可能是任何矩阵，P可能是任何矩阵。我们希望知道通过attention的操作，是否可以表达出所有合理的P（只需要列归一且为正）。

转换为证明：

$$
\forall X ,\forall P ; \exist\; W_k,W_q \\s.t. \;\;P=softmax(\frac{(W_kX)^T(W_qX)}{\sqrt{d_k}})
$$

假定$d_k=d_q=d$ , 我们可以证明

$$
case \;1:\;when \;d\geq n ,always \;\exist W_k,W_q,s.t.......\\case \;2:\;when \;d< n ,not\; \exist W_k,W_q,s.t.......
$$

在多头注意力的情况下，常规操作中，我们会减少$W_q,W_k,W_v$的维度来使得拼接后的结果和单头注意力得到的结果的维度相同，也就是说：

$$
Head(X)\in \R ^{\frac{d}{h}\times n}\\Multihead(X)\in R^{d\times n}
$$

此处$h$是头数（head size）。

显然的，从上述结论的角度来说，随着head size的变大，attention的表达能力会相应的有下降。这篇工作中的实验结果也说明了这一点(fig1 (a))。

![](/img1/Untitled%202.png)

但是multihead的提出实际上是为了提升attention的表达能力，也就是不能一味的削减head size。从另一个角度来说，通过提升embedding 的维度$d$或许可以避免这样的问题。

但是实际上，因为tokens数目很大，提升embedding维度会带来很大的计算和内存开销，又因为在对下游任务做微调时也是使用训练好的embedding矩阵，将更大的开销也带给了对所有下游任务的微调。

考虑到这些因素，这篇工作提出了 fixed multi-head attention ,即通过解除head size和$d_q,d_k,d_V$的关系，来实现在保证head size的同时，尽量避免降秩的问题。即就是：

$$
d_q=d_k=d_v=d_p\\FixedMultihead(X)\in \R ^{d_p h\times n}
$$

### seminar conclusion

decoder-only 中attention矩阵有mask操作，成为下三角矩阵。因为其行列式值非零，所以是满秩的。从此我们是否可以断言满秩的mask矩阵表达能力更强，使得decoder-only的架构好于encoder-decoder？

但是实际上我们可以发现，即使保证attention满秩，也不一定能做到矩阵P的完全表达。

同时，因为decoder-only只是单向注意力，而encoder-decoder在encoder部分是双向注意力。即使因为满秩单向注意力信息得到了充分的表达，也并不代表着比非充分表达的双向注意力信息更加全面。

综合考虑，从理论的角度并不能得出明确的结论。

### zero-shot for decoder-only

> Wang, T., Roberts, A., Hesslow, D., Le Scao, T., Chung, H.W., Beltagy, I., Launay, J. and Raffel, C., 2022, June. What language model architecture and pretraining objective works best for zero-shot generalization?. In *International Conference on Machine Learning* (pp. 22964-22984). PMLR.
> 

![](/img1/Untitled%203.png)

首先，模型有着不同的架构，如下图所示：
![](/img1/Untitled%204.png)

decoder：全部单向注意力

non-casual decoder: 输入部分是双向，输出部分是单向；但是输出和输入部分是共享参数的（也叫做prefix LM）

encoder-decoder：输入部分是双向，输出部分是单向；输入和输出部分不共享参数

同时，预训练阶段有着不同的任务：

![](/img1/Untitled%205.png)

这几种LM预训练目标更多的都用在生成任务上，而denoising作为预训练目标更多用于编码。

在此基础上，加上进行fine-tuning这一变量，对于各种变量进行自由组合并实验，可以得到如下结果：

![](/img1/Untitled%206.png)
![](/img1/Untitled%207.png)
![](/img1/Untitled%208.png)

从Finding 1 的角度，可以看出，decoder-only架构在做FLM时，不做fine-tune，zero-shot下表现最好。
