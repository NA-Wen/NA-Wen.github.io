---
title: Code-related Tasks Evaluation
date: 2023-09-05 18：37：20
index_img: /cover/3.jpg
tags: [NLP,code tasks]
author : NA-Wen
category: NLP
---
code-related tasks 有诸多，此博客解释目前使用较多的两种Eval：微软提出的CodeXGLUE + CodeBLEU以及OPENAI提出的 HumanEval
<!-- more -->
# Code-related Tasks Evaluation
## overview

- 最早的评价标准是BLEU ，基本思想是通过计算生成代码和标准答案代码的N-gram匹配程度。但是对于蕴含着丰富语法结构和语义信息的代码来说，此种方法显然无法足够准确的捕获关键的信息。
- 在此基础上，微软研究院提出了新的benchmark ，包含代码基准数据集CodeXGLUE ，以及评测标准 CodeBLEU。不过此种方法依然还是基于match，只是考虑了更多的语义和语法信息。
- 在此基础上，2021年，OPENAI提出了HumanEval，包含了164个编程问题，直接考察生成代码的正确性。其并非从现有源中查找，全部为手写。
- 同时，MBPP类似HumanEval，包含了974个编程任务，直接考察生成代码的正确性。

## CodeXGLUE + CodeBLEU

### tasks

CodeXGLUE 包含code-code,code-text,text-code,text-text四个类别（catagory）的任务；总共包含10个任务(task)，14个数据集(dataset)。如下图所示：

![](/img/Untitled.png)

具体来说，CodeXGLUE 中包含如下十项任务：

1. 代码克隆检测（Clone Detection）。该任务是为了检测代码与代码之间的语义相似度，包含两个外部公开数据集，但任务定义稍有不同。在第一个数据集中，给定两个代码作为输入，要求做0/1二元分类，1表示两段代码语义相同，0表示两段代码语义不同。在第二个数据集中，则给定一段代码作为输入，任务是从给定的代码库中检索与输入代码语义相同的代码。
2. 代码缺陷检测（Defect Detection）。该任务是检测一段代码是否包含可以导致软件系统受到攻击的不可靠代码，例如资源泄露、UAF 漏洞和 DoS 攻击等。该任务中使用了外部公开数据集。
3. 代码完形填空（Cloze Test）。先给定一段代码，但代码中的部分内容被掩盖住，该任务要求预测出被掩盖的代码。研究员们将该任务定义为**多项选择题**的形式，并构建了两个数据集。在第一个数据集中，被掩盖住的代码可以来自于代码中的任意字符；在第二个数据集中，则试图更有针对性地测试系统对代码 `max、min` 函数的理解能力。
4. 代码补全（Code Completion），也就是给定已经写好的部分代码。该任务能够自动预测出后续的代码，具体包含两个设置，分别是**词汇级别**（Token-level）和**行级别**（Line-level）的补全。顾名思义，前者的任务是补全下一个词汇，而后者的任务是补全一整行代码。词汇级任务使用了两个被外部广泛使用的数据。行级别的任务则是在词汇级别任务的数据上自动构建的数据。
5. 代码翻译（Code Translation）。该任务是把代码从一种编程语言翻译到另一种编程语言。研究员们构建了一个 Java 到 C# 的代码翻译数据集。
6. 代码检索（Code Search）。该任务是为了检测自然语言与代码之间的语义相似度，包含两个数据集，具体定义稍有不同：在第一个数据集中，给定一个自然语言作为输入，任务是从给定代码库中检索与输入自然语言语义最相近的代码，研究人员为该数据新构建了一个测试集，用来更好地测试系统的深层语义理解能力。在第二个数据集中，给定自然语言-代码对作为输入，要求系统做0/1二元分类，1表示语义相似，0表示语义不相似，研究员们同样为该任务构造了新的测试数据集，测试数据的自然语言来自必应搜索引擎，可以更好地反应真实用户的查询习惯。
7. 代码纠错（Code Refinement）。 给定一段有 bug 或者复杂的代码作为输入，该任务要求生成被优化后的代码。该任务中使用了一个外部公开的数据集。
8. 代码生成（Text-to-code Generation）。给定自然语言注释作为输入，该任务要求自动生成函数的源代码。该任务中使用了外部的公开数据集。
9. 代码注释生成（Code Summarization）。给定一段函数代码作为输入，该任务要求自动生成对应的自然语言注释。该任务中使用了外部公开数据集。
10. 文档翻译（Documentation Translation）。该任务的目的是自动将代码文档从一种自然语言翻译到另一种自然语言，如从英文翻译到中文。该任务中构建了新的数据集。

值得一提的是，微软提供了三种baseline：for understanding CodeBERT ; for generating CodeBERT+decoder ; for generating CodeGPT 。

### Method

CodeBLEU做评测的方法如下图所示，通过N-gram， weighted N-gram 以及对AST语法，Data-flow语义的match来做code-related任务的评估

![](/img/Untitled%201.png)

相比于BLEU，CodeBLEU更多了考虑了代码内部的信息，比如代码内部的关键字信息，AST语法树，以及对于data flow的match，最终的结果是此各项的加权求和；为了验证此指标的有效性，将CodeBLEU的结果和人类标注人员打分的结果进行皮尔逊相关系数计算，结果好于BLEU。但是此方法依然是基于match的评测方法。

## HumanEval

OPENAI在2021年推出了HumanEval，一个手写编程问题数据集，里面包含了164个手写的编程问题。每个问题包含：一个函数签名，文档描述，函数体和一些单元测试（平均每个问题有7.7个单元测试）。所有的问题和函数都是全手写，并不能在任何开源仓库中找到；有些model是基于github的数据集训练，训练数据中很可能已经包含了某些问题的答案源代码，因此此种方式保证了评测的公平和正确性。

HumanEval不是基于match的比较方法，而是通过测试unit test的方法来评判表现。

实际上，因为LLM 在生成内容上的随机性，如果对每个问题只生成一次，并不能充分反映LLM的能力。同时，我们还需要考虑LLM在生成内容上的稳定性如何。类似从均值和方差的两个角度，去评测LLM的总体能力和生成内容稳定性大小。

此处先解释LLM生成中的几个概念。

1. 解码策略：top-k，top-p
    
    在LLM 生成最后的结果时，每个可能的输出结果有其对应的似然分数。如果直接选择似然分数最高的结果，即为贪心解码。
    
    另一种常用的策略是top-k，从得分前k个tokens中抽样，使得其他的高分token可以被选中，很多时候可以提高生成的质量；k个样本中每个样本被抽样的概率根据其似然分数按比例分配。此方法称之为top-k。
    
    另一种类似于top-k的算法top-p，根据似然分数得分从上至下，将可能性之和不超过p的top tokens 加入到抽样名单中，每个样本被抽样的概率根据其似然分数按比例分配。
    
    如果 *k* 和 *p* 都启用，则 *p* 在 *k* 之后起作用。
    
2. temperature（0：）
    
    因为抽样策略的采用，因此LLM生成结果具有随机性。通过调节temperature可以影响此种随机性的大小。较低的温度（0.2，0.3算小的）意味着较小的随机性/更加集中的分布，生成的内容更容易在training data中观察到范式；较高的温度（0.8 ，0.9算高）则相反。
    
    温度为0时始终输出相同的结果。在针对不同问题进行调节时，温度为1通常是一个比较好的起点。
    

在评测LLM 的生成结果时，通过让LLM生成多次结果，计算其至少通过一次的概率，称之为$pass@k$。如下图：k越大，通过率越高；温度越高，斜率越大。

![](/img/Untitled%202.png)

有几点需要说明：

1. $pass@k$可以看作是一个新的指标：在k次测试中如果至少有一次通过，即为1；如果都未通过，即为0。相对于只做一次测试来说，是放宽了条件。
2. $pass@k$并非从概率上对于LLM 生成内容的通过率进行的衡量，而只是频率的表征。根据大数定律，只有在试验次数 $t$ 足够大时，$pass@k$才会接近真正的通过概率。

在测算每一次 $pass@k$时，需要做$kt$次实验，最终加起来的次数会过于多了。因此在提出此标准时，OPENAI也提供了一种无偏估计的方法。

$$
pass@k=E_{Problems} (1-\frac{
\begin{pmatrix}
n-c \\
k  \\
\end{pmatrix}
}{\begin{pmatrix}
n \\
k  \\
\end{pmatrix}
})
$$

其中，$n$是总试验次数，$c$是这$n$次中成功的次数。

$$
\frac{
\begin{pmatrix}
n-c \\
k  \\
\end{pmatrix}
}{\begin{pmatrix}
n \\
k  \\
\end{pmatrix}
}
$$

上式中，分母指从n次中选取k次的选法，分子指从失败的次数中选取k次的选法。即在已经做完n次实验的基础上，选取k次结果，k次结果全部为失败的概率。

如果用每次生成代码的通过率p来进行对上式的估计，即就是认为c符合二项分布，得到的最终结果为：

$$
pass@k=1-(1-p)^k
$$

比较此式计算出的结果，可以发现此种估计是有偏的。

![](/img/Untitled%203.png)

因此只能对上式在不同的c的情况下，一项一项进行计算，最后算出期望，这样的结果实验验证出来也是无偏的。

![](/img/Untitled%204.png)

总结来看，pass@k的结果来自于：做n次试验，记住n次中成功的次数c，即可得到所有的pass@k值，但是此时还是估算出的频率；多次重复此过程并计算期望，即可得到估算出的pass@k概率。

## MBPP

****MBPP： Mostly Basic Python Programming****

包含974个python的编程问题。每个问题包含：文字描述、对应代码、unit test （平均每个问题3个）（ `'text', 'code', 'test_list', 'test_setup_code', 'challenge_test_list'` ）

reference :

[代码智能新基准数据集CodeXGLUE来袭，多角度衡量模型优劣 (msra.cn)](https://www.msra.cn/zh-cn/news/features/codexglue)

[openai/human-eval: Code for the paper "Evaluating Large Language Models Trained on Code" (github.com)](https://github.com/openai/human-eval)

[代码生成模型评价指标 pass@k 的计算 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/653063532)

[MBPP Dataset | Papers With Code](https://paperswithcode.com/dataset/mbpp)