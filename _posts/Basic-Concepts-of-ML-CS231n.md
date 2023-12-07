---
title: Basic knowledge of machine learning (1) 
date: 2023-07-21 14:14:18
category: MLBasic 
author : NA-Wen
tags: [linear,loss]
index_img : /img/linearpost.png
---
此chap有关基础的ml知识和概念，包括naive knn,linear layer(loss function,regularization, optimization and backpropagation),同时也介绍了部分激活函数。
<!-- more -->
**基于CS231n(Stanford computer vision)课程，整理笔记。**
# image classification
For computer , there are some problems in recgnize a picture , here are a list :
视角变化（Viewpoint variation）：同一个物体，摄像机可以从多个角度来展现。
大小变化（Scale variation）：物体可视的大小通常是会变化的（不仅是在图片中，在真实世界中大小也是变化的）。
形变（Deformation）：很多东西的形状并非一成不变，会有很大变化。
遮挡（Occlusion）：目标物体可能被挡住。有时候只有物体的一小部分（可以小到几个像素）是可见的。
光照条件（Illumination conditions）：在像素层面上，光照的影响非常大。
背景干扰（Background clutter）：物体可能混入背景之中，使之难以被辨认。
类内差异（Intra-class variation）：一类物体的个体之间的外形差异很大，比如椅子。这一类物体有许多不同的对象，每个都有自己的外形。

**data-driven**
so traditional algorigthm can not solve the problem like solving a sorting problem .we design a data-driven algorithm to do this task( robust matters ).
here are a complete process for this algo:inclding generate a model ,and using the model to classify.(packed in a python class )

输入：输入是包含N个图像的集合，每个图像的标签是K种分类标签中的一种。这个集合称为训练集。
学习：这一步的任务是使用训练集来学习每个类到底长什么样。一般该步骤叫做训练分类器或者学习一个模型。
评价：让分类器来预测它未曾见过的图像的分类标签，并以此来评价分类器的质量。我们会把分类器预测的标签和图像真正的分类标签对比。毫无疑问，分类器预测的分类标签和图像真正的分类标签如果一致，那就是好事，这样的情况越多越好。

Given a set of images and labels , we need to complement a algo to classify more image. 
## knn
A naive thinking is that we  memorize all the images and labels given , and when facing a new image , we find the likest image in what we learnt .That's  the insight of Knn.  
1. how to evaluate the "likest":
   1. L1 disatnce : the sum of all difference of each pixels 
   2. L2 distance :the sqrt of sum of all difference ^2 of each pixels
   here the L2 is better than L1 ,when changing the coord does not change the results .
2. Why k? not 1 
   1. better performance on generalization . more smmoth boudary .
   [http://vision.stanford.edu/teaching/cs231n-demos/knn/]  
   2. how to determine which k brings to the best performance in classifying ?
        we can adjust some hyperparameters to get the  best  model ,but which part of  data should be used?Test data is hidden during the whole training process ,the ability of a model is how it performances when facing unknown problem . But we can divide some of the train data to adjust the mdoel , called **validation** set. 
        The insight is that we let the model do pre-test /fake-test on the data it had met .It is simpler for the model. Like the test paper cannot be exposed to students ,the test data cannot be used to the model until the model para fixed.  

**Advantages**:
$\Theta(1) $in training (easy to understand) 
**Disadvantages:**
$\Theta(n) $in testing ,which is the complexitity we really care about .
No semantic meaning is captured in the distance : we can use some tricks to achieve large distance (like shift a picture or add a filter ),but actually the body in the image keeps the same. This leads to that if we want more accuracy ,we need to a dense enough dataset.
## linear layer
**什么是线性分类器**
线性：线性函数；分类器：函数实现分类的效果
1. 首先，对图片的处理实际上是对每一个pixel的值的处理；将其排成一个向量x，W作为权重,b作为偏置,f是我们使用的分类函数。
   $$f=Wx+b$$
此处注意，把b concat到W矩阵的最后一列，同时把pixel向量x的维度补一个零，这样只用在后续学习一个矩阵，无需再多学习一个偏置的向量。
2. 从图片的视角来看：如果将每一个图片是做一个高纬度的点，做linear classification实际上是在图片的高维空间中做了线性划分，划分出了每个图片类别。
   另一种看法是，线性分类器可以学习到每个图片类别的一个template：因为计算f的过程实际上是做w和某图片向量的内积；求得到最能代表属于某一类的f值，此时会有一个W和对应可求得此f值的pixel向量x。因此说这个W反应的是学习到的某个template(体现为向量x)，即此时的pixel向量转换成的图片。
>NOTE：在预处理时需要对图片信息做零均值的中心化处理(减去平均值再进行归一化，分布在[-1,1]之间)

## loss function
From above, we have a score function called f. But how we evaluate the performance of the score function? 
We need to consider the initial label for each image(maybe tagged by human ? real category), and compare them with the $f$ classification result .By calculating the difference ,we can evaluate the performance of the current $f$ .
### SVM loss
 From SVM(支持向量机) , we can get **L1 loss function** .
 $$L=\frac{1}{N}\sum_{i}L_i$$
 $$L_i=max(0,s_i-s_{y}+1)$$
 Here, $y$ is the label of the input image . $s$ is the result of f,$s_i$ meaning score for the $i-th$ class.And $i!=y$.
 This loss function is called L1 loss function , it punishes the $s_i$ which is larger than ($s_y-1$)
 也叫折叶损失(Hinge loss)
 我们在此处关注的是错误分类的得分和正确分类的得分之间的高低差值(是错误分类得分减去正确分类得分)，如果差值比较大(此处是大于1)，则惩罚此得分。

 **L2 loss funtion** :
   $$L=\frac{1}{N}\sum_{i}L_i$$
 $$L_i=max(0,s_i-s_{y}+1)^2$$
This loss function punishes  more strongly.
### softmax 
We consider each result a non-normalize and log probability of belonging to a certain catagory. So we define a loss function ,which only take the probability of belonging to the correct class into account.
$$L_i=-log(p_y)$$
$$p_y=\frac{e^{s_y}}{\sum_{i}e^{s_i}}$$
If the p is larger , the loss will be smaller . In contrary , p is smaller , loss is larger.
>NOTE:here $e^s$ might be large , so we can do s  = s - max(s) first , this will not change the p , but keep the stablity of computing. 

>从概率论的角度来理解，我们就是在最小化正确分类的负对数概率，这可以看做是在进行最大似然估计（MLE）。该解释的另一个好处是，损失函数中的正则化部分R(W)可以被看做是权重矩阵W的高斯先验，这里进行的是最大后验估计（MAP）而不是最大似然估计。
### regularization
Obviously  ,we can notice that there might be more than one W but we can get a same f result . Which W is better? Besides, if W times 2, the loss result will increase (in SVM).
So the loss function contains two components :
$$Loss \; function \;= \;data \;Loss \;+\; regularization\;Loss$$  
$$regularization\;Loss\;=\;\lambda R (W)$$ 
L1 :$ R(W) = \sum\sum W$ 

 L2: $R(W) =  \sum\sum W^2$(more common)

The regularization first punishes too large W(max margin property) ,and advance the generalization ability of f, for focusing on each feature instead of only focusing on several features. 
> NOTE:here only consider $\lambda$ is OK, no need to consider the $\Delta$(set as 1.0 in SVM) , the same effect.

更进一步的，正则化对解决过拟合提供了思路。规范化的神经网络泛化的性质会好于非规范的神经网络，也就是说不容易发生过拟合，而前者更容易出现较小的权重，因此损失函数加上正则化项则是更倾向于权重较小的网络。
### difference between SVM and softmax
SVM 的值是无标定的，做差比较不同分类之间的值才有意义；但是Softmax的值本身代表着概率/可能性。
在使用上，Softmax更加的local objective，只要正确分类的概率没有到1，loss的值就不为0，即不断地希望求到更大的正确分类的可能性；但是SVM 只要保证在一定的空间范围内，其损失值都是0 . 

## optimization
We want to get the minimum value of loss result , which represent the best performance of scroe function f. So how we adjust W to decrease the loss function faster? 
1. random : we random choose a list of W and calculate the loss 
   better than randomly classify ,but long way to SOTA
2. Follow the gradient 
   We decreases the W in the gradient direction.也就是loss对loss function中W求梯度，W顺着梯度方向的反方向走，可以使得loss的取值下降最快。
   -> better 
   For a larger amount of samples , we need do calculate loss for each sample , then calculate gradient for each sample, costs too much.
   SGD : the naive process costs a lot ; we can use a minibatch instead to calculate the loss and the gradient. Then change W.  
   $$   W-=step\_size\times W_{grad} $$
   here the step_size is also called learning rate ,which is set on your own .
**公式示例如下：**
以单层为例，激活函数为$f(z)$,损失函数为$C$
$$z=w^Tx+b$$
$$y=f(z)$$
$$C=\frac{1}{n}\sum_x C_x
=\frac{1}{n}\sum_x ||y-a||^2$$
$$\Delta C=\frac{\partial C }{\partial w}\Delta w,\:\:\Delta C=\frac{\partial C}{\partial b}\Delta b$$
$$\Delta w=-\eta \frac{\partial C }{\partial w},\Delta b=-\eta\frac{\partial C }{\partial b} :可保证C值减少$$
以此种方式更新权值与偏置使得损失函数的值下降的方式叫做梯度下降。
按照上述方式更新参数，每一次需要对每一个(共$n$个)样本都求梯度。在输入的样本数量很大时，此方法耗时过长。随机梯度下降可以有效改善这个问题：即在每一次要利用梯度下降更新参数时只随机的去取总样本中的$m$个样本$(m=mini\_batch\_size)$
$$ \Delta w=- \frac{\eta}{n}\sum_x \frac{\partial C_x}{\partial w}$$
$$\Rightarrow \Delta w=- \frac{\eta}{m}\sum_{j=1}^{m} \frac{\partial C_{X_j}}{\partial w}$$
在每一个epoch中，以上述的形式去取mini_batch，取完所有的样本即一个epoch结束，要注意选取的随机性。

>NOTE:怎么算梯度？
使用有限差值近似计算梯度比较简单，但缺点在于终究只是近似（因为我们对于h值是选取了一个很小的数值，但真正的梯度定义中h趋向0的极限），且耗费计算资源太多。
第二个梯度计算方法是利用微分来分析，能得到计算梯度的公式（不是近似），用公式计算梯度速度很快，唯一不好的就是实现的时候容易出错。为了解决这个问题，在实际操作时常常将分析梯度法的结果和数值梯度法的结果作比较，以此来检查其实现的正确性，这个步骤叫做梯度检查。

## backpropagation
In fact , backpropagation is a process of deravation. Here we lead to one important thing: "gate".
A gate can gets several input, and output something . A gate represents a function.We need to do derivation to all the inputs , so we need to know the function and the input.
For example :
$$f=wx$$
$$\frac{\partial f}{\partial x}=w, \frac{\partial f}{\partial w}=x$$
f is one gate .If we recursively call the same process for each gate ,and use chain rule to multiply *->* the process is called backpropagation.
>NOTE:here we donot care about which input is "variable", which is "parameter". all are inputs , we can do derivation to all of them.(But only those donot related to the initial input in number matters)

>NOTE: caffe is a popular but old framework which contains many "gate" module , each of which has a forward and a backward function

>NOTE: If you have difficulty in high dimension problem , try to reduce dimensions and then do it.

## 激活函数
基本的几种激活函数
1. sigmoid 
   $$f(x)=\frac{1}{1+e^{-x}}$$ 
2. ReLU
   $$f(x)=\left\{\begin{aligned}
x&  & x> 0 \\
0& &x\leq 0 
\end{aligned}
\right.$$
3. tanh(x)
   $$ f(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$
4. leaky ReLU
   $$f(x)=\left\{\begin{aligned}
x&  & x>0 \\
\alpha x&&x\leq 0& (\alpha 很小)
\end{aligned}
\right.$$
5. Exponential Linear Unit
   $$f(x)=\left\{\begin{aligned}
&x  & x>0 \\
& \alpha (e^x-1) &x\leq 0 
\end{aligned}
\right.$$

Q1：为什么要使用激活函数？
为了给神经元增加非线性的输入
Q2：区别？
sigmoid 是最早的激活函数(模仿生物体神经元)，输出值在(0,1)之间使之可以用于二分类。tanh函数相比sigmoid函数在0附近更加陡峭，所以在训练前期速度比较快。
以上两种都是饱和激活函数(对数据进行了压缩)，在输入值较靠近上下限时，导函数值趋于零。因此在训练到后期时，输出结果很靠近上下界中某一边，此时反向传播因其对激活函数的求导项趋于0，因此权值改变慢，学习速度放缓。
ReLU可以有效改善此种饱和的情况，对于正值不会饱和，训练速度很快。但是一旦其输入出现负值，反向传播时对激活函数的求导值为0，相关权值就不会再改变，该神经元死亡。
Leaky ReLU和ELU都解决了ReLU中神经元死亡的问题，即对小于零的输入导数值不为0。此两者相比，前者因为不涉及指数运算，所以运算速度较快，后者实际学习效果较好，且后者在0附近平滑(此点类似tanh)，可以加速收敛。
>注意，因为sigmoid不是0为均值的，所以最好不要出现在隐藏层里，最好放在输出层。
不要陷入思考的细节，只用考虑激活函数的本质是映射，只用想讲什么范围映射到什么范围去

## conclusion
激活函数是线性分类器到神经网络的一个big gap，为f增加了非线性的成分。













