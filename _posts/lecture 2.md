---
title: Basic knowledge of machine learning (3) 
date: 2023-09-25 23:31:11
index_img: /img/linearpost.png
tags: [nomalization,MLE,math]
category: MLBasic 
author : NA-Wen
---
回答了机器学习过程中三个基本问题(1. what is the hypothesis function? 2. what is the loss function? 3. how do we solve the training problem?). 包含了部分数学证明。
<!--more-->
***This note is compiled based on the course materials provided by Professor Mingsheng Long in course "machine learning" at tsinghua university.*** 

every machine learning algorithm should solve three basic problem :

1. what is the hypothesis function?
2. what is the loss function?
3. how do we solve the training problem?

# An example : predict the peak power consumption in summer

- first do **feature engineering** : which attribute is x ,which is y
- why called “regression”? label space   $y=R$
- EDA (exploratory data analysis ) to ease model selection
    - done before you choose model
    - not a function ? more x attributes needed(需要更多的自变量/feature)
- linear regression
    - feature vector : $x\in R^{d}$, label : $y\in R$
    - whole hypothesis space **h**
        
        ![Untitled](/img3/Untitled.png)
        
        (w corresponds to each high-dim plane in hypothesis space)
        
    - loss
        - use the squared loss(L2 loss)
            
            ![Untitled](/img3/Untitled%201.png)
            
    - process
        
        ![Untitled](/img3/Untitled%202.png)
        

Until here , we have solved the first two problems , then comes to a new problem : how we solve the training problem? 

## optimization

- how to compute w matrix?
    - Analytic solution?
        
        $$
        data \;matrix:\;X=\begin{bmatrix}--x_1^{T}-- \\ --x_2^{T}-- \\ --x_3^{T}-- \\ . \\ . \\ . \\ --x_n^{T}--
        \end{bmatrix}
        $$
        
        $$
        label\;matrix:\;y=\begin{bmatrix}y_1 \\ y_2 \\ . \\ . \\ . \\ y_n\end{bmatrix}
        $$
        
        $$
        \hat{\epsilon}=\sum_{i=1}^{n}(w^Tx_i-y_i)^2 \\ =||Xw-y||^2 \\ =(Xw-y)^{T}(Xw-y) \\ =w^{T}X^{T}Xw-w^{T}Xy-y^{T}Xw+y^{T}y
        $$
        
        $$
        \frac{\partial\hat{\epsilon}}{\partial w}=\frac{\partial w^{T}X^{T}Xw}{\partial w}-\frac{\partial w^{T}X^{T}y}{\partial w}-\frac{\partial y^{T}Xw}{\partial w}+0 \\ =2X^{T}Xw-2X^{T}y
        $$
        
        $$
        w=(X^{T}X)^{-1}X^{T}y
        $$
        
        the complexity is unaffordable : $O(d^2(d+n))$ , and always more dim in features when using linear model 
        
    - so instead , we do optimization !
- optimization
    - (for differentiable loss function) GD (gradient descent): follow the steepest slope , *but you only observe local information for each step*
        - gradient :
            
            $$
            g=\nabla J(w)  \\ g_j=\nabla_jJ(w)
            $$
            
            According to taylor approximation :
            
            $$
            J(w)=J(w_0)+(w-w_0)^Tg+...
            $$
            
            so we can get : 
            
            $$
            J(w)=J(w_0-\eta g)-\eta g^Tg
            $$
            
            different order taylor approximation → different order optimization method (one-order :faster ; two-order : more accurate )
            
            这里用的是一阶泰勒展开对应的优化方法
            
        - iteration
            - do until converge : T
                
                $$
                w^{t+1}\leftarrow w^t-\eta(2X^{T}Xw-2X^{T}y) 
                $$
                
            - learning rate $\eta$  matters
                
                ![Untitled](/img3/Untitled%203.png)
                
            - complexity $O(dnT)$
            - Theoretically, convergence rate is $\frac{1}{\sqrt T}$ under convex condition.(can be proved)
    - SGD (stochastic gradient descent)
        - randomly sample a small batch (size of m)
            
            $$
            w^{t+1}\leftarrow w^t-\eta(2X_m^{T}X_mw-2X_m^{T}y) 
            $$
            
        - complexity $O(dmT)$

So we know how to complement the training process , but why do we use L2 loss in the process? Now give a proof / explanation from statistical view.

## statistical : MLE

First, supposed we have a parametric model $\{p(z;\theta)|\theta\in \Theta\}$ 

- **( assumption 1 )** $D=\{z_1,z_2,...,z_n\}$独立同分布

$$
p(D;\hat{\theta})=\prod_{i=1}^np(z_i;\hat{\theta})
$$

due to numerical stability , we prefer to work with log-likelihood 

$$
log (p(D;\hat{\theta}))=\sum_{i=1}^nlogp(z_i;\hat{\theta})
$$

Maximum Likelihood Estimator (MLE) for estimating the parameter
 $\theta$ in the parametric model$\{p(z;\theta)|\theta\in \Theta\}$

$$
\hat{\theta}\in argmax_{\theta\in \Theta}\sum_{i=1}^nlogp(z_i;\hat{\theta})
$$

The MLE leads to a particular loss function.

We all know in supervised learning , one $z$ contains $x,y$(feature ,label) ; so the p can be

$$
\prod_{i=1}^np(z_i;\hat{\theta})=\prod_{i=1}^np(y_i|x_i;\hat{\theta})p(x_i)
$$

Getting $p(x_i)$ is difficult (in fact for generation model we get this), so we do another assumption here.


- **( assumption 2 )** 对样本的分布认为是经验分布，但是这种假设缺乏对于样本形状的先验假设，因此对于对抗样本鲁棒性变弱。


<aside>
💡 "经验分布" 是一个统计学和数据分析领域的术语，通常指的是根据已经观测到的数据样本来估计一个随机变量的概率分布。它是一种非参数估计方法，不依赖于对分布形状的先验假设。
经验分布通常通过将数据样本中的每个观测值视为一个概率质点来表示。这意味着每个观测值都有一个与之相关的概率质点，概率质点的高度（或权重）等于该观测值在样本中出现的频率。

在数学符号中，假设有一个包含观测值的数据样本集合 $x_1, x_2, \ldots, x_n$，那么这些观测值的经验分布可以表示为：

$$
\hat{F}(x) = \frac{1}{n} \sum_{i=1}^{n} I(x_i \leq x)
$$

</aside>

$$
\prod_{i=1}^np(z_i;\hat{\theta})=\prod_{i=1}^np(y_i|x_i;\hat{\theta})p(x_i)=\prod_{i=1}^np(y_i|x_i;\hat{\theta})\frac{1}{n}
$$

We can only focus on the  conditional probability.

$$
\hat{\theta}=max_{\theta\in \Theta}\sum_{i=1}^np(y_i|x_i;\hat{\theta})
$$

- **( assumption 3 )** add gaussian noise assumption: 因为观测中可能有噪音，假设噪音服从高斯分布
    
    $$
    y=w^Tx+e \\ e\sim\N(0,\sigma^2)
    $$
    

Adding the noise does not change the expectation. 

![Untitled](/img3/Untitled%204.png)

Then the y should also belong to a gaussian distribution

- **( assumption 4 )** linear model

$$
y\sim \N(w^Tx,\sigma^2)
$$

$$
\frac{1}{n}\sum_{i=1}^nlogp(y_i|x_i;\hat{\theta}) \\ =\frac{1}{n}\sum_{i=1}^nlog(\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2)) \\ =\frac{1}{n}(\frac{n}{2}log(\frac{1}{2\pi\sigma^2})-\sum_{i=1}^n\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2))
$$

Until here , we know maximum Likelihood Estimation of w under the Gaussian noise assumption is equivalent to linear regression with the squared loss.

*long-tailed distribution :对于噪声更加鲁棒；在分布中已经考虑进来了 → 此种情况下损失函数长什么样？*

![Untitled](/img3/Untitled%205.png)

![Untitled](/img3/Untitled%206.png)

# Nonlinearization

- feature map  x→z
    - basis function : the mapping function, non linear transform
    - polynomial function
    - **RBF (radial basis function)**
        
        ![Untitled](/img3/Untitled%207.png)
        
    
    将原本的特征做了映射，达到了非线性的效果。做多次不同的映射可以获得特征上的多样性。但是在回归过程中还是做的线性预测。整体上提高了模型的拟合能力，拥有了拟合非线性函数的能力。
    
    依然需要EDA来做特征的选择！
    
- local weighting
    - 将局部都是视为线性的函数，用局部附近的信息去预测局部
    - 和transformer的思想类似 ? 通过作query和key的匹配（相似度高相当于x距离很近），来对于value加权。认为和当前token更相似tokens的对于当前token的embedding更重要。

在引入了非线性模型之后，虽然模型的拟合能力增强了，但是也更容易出现过拟合的问题。我们通过正则化（正规化）来解决。

# regularization

how to deal with overfitting ?

- an observation
    
    when overfitting appear , the weight parameters are too large 
    
    belike :
    
    ![Untitled](/img3/Untitled%208.png)
    
    So constraining the size of parameters is a direct way to escape from overfitting. 
    
- L1 L2 Norm
    - L2 regularized linear regression (ridge regression) :commonly used
        
        ![Untitled](/img3/Untitled%209.png)
        
        equals to :
        
        ![Untitled](/img3/Untitled%2010.png)
        
        <aside>
 
        💡 拉格朗日乘子法是一种用于求解带有约束条件的极值问题的方法，通常用于优化问题。下面是使用拉格朗日乘子法求解极值的一般步骤：
        假设你有一个带有约束条件的优化问题，要最大化（或最小化）函数 $f(x, y, \ldots)$
        
        目标函数：$f(x, y, \ldots)$
        
        约束条件：$g(x, y, \ldots)=c$
        
        其中 $x, y, \ldots$是你要优化的变量，*c* 是常数。要使用拉格朗日乘子法求解这个问题，你可以按照以下步骤进行：
        
        1. **建立拉格朗日函数：** 定义一个新的函数，称为拉格朗日函数（Lagrangian），它是目标函数和约束条件的线性组合，带有一个额外的参数（拉格朗日乘子）：
            
            $$
            L(x, y, \ldots, \lambda) = f(x, y, \ldots) - \lambda \cdot (g(x, y, \ldots) - c)
            $$
            
            这里，*λ* 是拉格朗日乘子。
            
        2. **计算拉格朗日函数的偏导数：** 对拉格朗日函数 $L(x,y,…,λ)$ 分别对 $x,y,…,λ$求偏导数，并令它们等于零：
        3. **解方程组：** 以找到最优的  $x,y,…,λ$。这些值将给出目标函数在满足约束条件下的极值。
        </aside>
        
        ![Untitled](/img3/Untitled%2011.png)
        
    
    when $λ$ is larger , the red part is more ?
    
    - L1 : better for sparse solution
        
        ![Untitled](/img3/Untitled%2012.png)
        
        ![Untitled](/img3/Untitled%2013.png)
        
    
    参数的稀疏性是衡量模型复杂度的重要指标，正则化的操作(图像上看是有norm尖点)可以有效地在此点上予以改善。
    
    - L0
        
        ![Untitled](/img3/Untitled%2014.png)
        
        ![Untitled](/img3/Untitled%2015.png)
        
        ![Untitled](/img3/Untitled%2016.png)
        
    - L1
        
        ![Untitled](/img3/Untitled%2017.png)
        
        ![Untitled](/img3/Untitled%2018.png)
        
        ![Untitled](/img3/Untitled%2019.png)
        
    - L2
        
        ![Untitled](/img3/Untitled%2020.png)
        
        ![Untitled](/img3/Untitled%2021.png)
        
        ![Untitled](/img3/Untitled%2022.png)
        
    
    三种正则化操作都是通过限制参数的复杂程度来限制模型的复杂程度。不同的设置导致参数的受限情况以及导致的参数复杂度的衡量不同。但是L2对于参数稀疏化的效果比不上前两个（更多是smooth optimization）；而L0相比于L1又是非凸的。
    
    Why regularization works? We can explain it through Lipschitz constant (🤩 really an amazing solution! )
    
    $$
    lipschitz\; theorem:D(f(x_i),f(x_j))\leq C* D(x_i,x_j)
    $$
    
    $$
    D(f(x_i+\Delta),f(x_i)) \\ =D(w(x+\Delta),wx) \\  =||w\Delta||_2^2 
    $$
    
    So C here is $||w||_2^2$
    
    对于其他的范数，此种证明也是合适的。都可以得到参数$w$的范数作为Lipschitz常数出现。当函数自变量变化比较小时，如果f值变化也较小，此时函数一般具有更好的光滑性和更慢的变化速率，这样的函数一般来说是更合意的。
    
    这就解释了为什么我们有根据范数导出的不同正则化方法。