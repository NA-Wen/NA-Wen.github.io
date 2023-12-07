---
title: Transfer C++ into Python Module
date: 2023-08-23 9:51:11
index_img: /cover/1.jpg
author : NA-Wen
tags: [C++,Python]
category: TechBasic 
---
>这是大一的OOP大作业，等code结果时翻出来了。虽然有诸多不求甚解的地方，但是写的比较细致也或许有点作用吧。放上来博客来。
<!--more-->




**将C++移植成Python类库，使用pybind11，ctypes或者Cython，为某次小作业添加Python接口，并编译成Python库。编写Python程序调用得到的库，实现和小作业相同的输入输出功能。讲解相关的配置、编写方法。**








## PART  1  Cython

### 一、前置知识与环境配置

Cython  为python编程语言编写C的扩展，从而提高效率，cython语言被称作python的超集。Cython语言编写的文件后缀为.pyx；语法是C与python的混合（其实是在python语法的基础上添加了少数关键字以此利用C的特性，以此实现对C的轻松访问）

> *Note ：此处要注意与cpython的区别：cpython是C语言开发的解释器，在运行python时就要调用这个解释器。解释器将源代码转换成字节码，再转换成机器语言与操作系统进行交互。*
>
> 

**本机环境：windows10 | python 3.9 | MSVC  使用VScode编写**

#### 前期环境配置

1. python解释器的下载，选择了3.9版本（注意把python添加到环境变量）

2. vscode中python开发环境的配置
   1. 安装python的扩展，选择安装即可
   2. 在command palette中选择python解释器的路径

3.  pip镜像源

​       			安装包cython时需要pip，在不使用镜像源安装时，即使运行shadowsocks，也会出现安装的问题，因此配置镜					像源，选择了清华的tuna镜像源

​							新建pip文件夹，在其中新建pip.ini文件,文件内具体配置如下：

```makefile
[global] 

index-url = https://pypi.tuna.tsinghua.edu.cn/simple 

[install] 

trusted-host = https://pypi.tuna.tsinghua.edu.cn  
```

此处环境本机配置截图如下：

<img src="/img/a0ae2c993eb2810dff8c4f97a3caac0.png" alt="image-20220606130256463" style="zoom:50%;" />



 

### 二、具体实现

#### 1.以某次小作业的“求最大公因数 和 最小公倍数”为例，代码实现

可以写出python代码

```python
def gcd(a,b):
    if b!=0:
        return gcd(b,a%b)
    else:
        return a

def gc(a,b):
    return (a*b)/gcd(a,b)
```

cython代码

```cython
from libc.math cimport pow

cpdef long long int test(long long int x,long long int y):
    cdef long long int a=x
    cdef long long  int b=y
    if b!=0:
        return test(b,a%b)
    else:
        return a

cpdef long long int test1(long long int x, long long int y):
   cdef long long int ans
   ans=(x*y)/test(x,y)
   return ans 
```

对比这两段代码，可以发现几乎差不多

不同之处：

1. Cython程序扩展名为.pyx

2. cimport与import不同

3. Cython中的函数用cpdef定义，而且写了函数返回值的变量类型，在声明变量时也出现了cpdef

#### 2.编译

Cython程序需要编译后才可以被python调用，具体过程是：Cython编译器（事实上也是Cpython），编译成调用python源码的C/C++代码，再编译成dll

即：需要再编写setup.py，来实现上述过程

```python
from distutils.core import Extension, setup 

from Cython.Build import cythonize 
# define an extension that will be cythonized and compiled 

ext = Extension(name="hello", sources=["hello.pyx"]) 

setup(ext_modules=cythonize(ext)) 
```

这里的name ，是编写的.pyx文件名。

> 这是一个写的比较简单的setup.py，其中还有许多参数可以被指定（但此处没有必要进行指定），比如说：改变传给c的编译器的-I ，-L参数。

之后在vscode的终端执行下述命令：

```
python setup.py build_ext --inplace
```

运行结果如下：

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132744158.png" alt="image-20220606132744158" style="zoom:50%;" />



#### 3.调用

最后一步，用python调用编译好的动态链接库

```python
import hello
```

我们进行100000次计算，输出python与调用cython计算的时间代价

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132759608.png" alt="image-20220606132759608" style="zoom:50%;" />

可以看到明显的时间开销上的差距，基本相差了20倍的效率。

### 三、一些思考

Cython提供给我们一个很好的工具，在相应的文件夹下

```python
cython -a  hello.pyx
```

会生成一个名为hello的html，里面黄色标出的就是与python发生交互的地方 ，黄色越深说明发生的交互越多。

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132813581.png" alt="image-20220606132813581" style="zoom:50%;" />

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132825438.png" alt="image-20220606132825438" style="zoom:50%;" />

可以看到，其中使用关键字的地方即没有与python发生交互，大大减少了时间的开销.

调用python这样一种解释语言的时候会产生额外的开销，即调用解释器产生的额外开销。解释器按照读取高级程序的语句 ，转换为某种中间表示形式，然后进行“解释”。Python首先要经过编译的步骤，讲源代码转换为python字节码，字节码被缓存保留在pyc文件当中。

这里用dis进行反编译看一下上述demo.py的字节码，下是一小部分的截图 

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132841737.png" alt="image-20220606132841737" style="zoom:50%;" />

可以看到字节码文件装载了一系列的指令。Cython再从python源代码编译出一系列python字节码之后，再通过python的虚拟机（VM）去执行字节码里的指令。VM实际上是对物理CPU的模拟，但是是完全基于栈数据结构的。VM将指令转换成可以被操作系统以及CPU理解并执行的低级操作。此处与本机编译上有时间差距，会慢一点。

与C对比，C没有这种解释器，虚拟机，也不用被翻译成高级字节码，直接编译器将其编译成为机器代码，直接由CPU运行。

Cython减少时间开销就是借助了此种区别。如果将Cython的源代码转换成一致的拓展模块，即已编译的模块，就没有了解释器的性能时间开销。

 

## **PART  2 **pybind11

### 一、前期知识与环境配置

**本机环境：VS2019 |  Windows 10 |   Python 3.9**

Pip与python下载等相关操作在PART  1已经完成，此处主要说明前期需要配置的相关VS环境。

1. 将pybind11 集成到项目

​				在vcpkg中安装pybind11，之后利用nugets将其配置到项目当中，即可通过引用头文件来使用使用vcpkg（开源的 				命令行的包管理工具，并不依赖VS，同时直接下载源码，避免了二进制不兼容的问题）

具体步骤如下：

​						1.从github上clone vcpkg仓库，然后执行安装命令

​						2.用vcpkg进行包安装与部署

打开powershell界面，输入

```
.\vcpkg install pybind11:x86-windows
```

之后将其部署到项目

```
.\vcpkg integrate project
```

本机配置截图如下：

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132902560.png" alt="image-20220606132902560" style="zoom:50%;" />

2. 配置VS环境
   1.  在VS上安装python开发负载，以及安装python （3.9版本），在PythonApplication 下的python环境中，进行选择。
   2. 核心的项目是以空项目的形式创建的，源文件中添加.cpp文件，逐步写出扩展模块

  因此需要修改的项目属性如下：

```yaml
   1. 常规--高级--目标文件扩展名  修改为.pyd

   2. 常规--配置类型  修改为动态库（.dll）

	3. VC++ 目录--库目录--添加libs的路径（**此处务必注意**，是libs，下属文件属性是.lib，不是Lib，下属文件为.py）

   4. C/C++--常规--附加包含目录  添加include文件夹的路径

   5. C/C++--预处理器--预处理器定义  _DEBUG改为NDEBUG

   6. 链接器--附加库目录 添加libs的路径

   7. 链接器--输入--附加依赖项 添加python39_d.lib 
```

本机修改后的配置截图如下：

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132920965.png" alt="image-20220606132920965" style="zoom:50%;" />



### 二、具体实现

将c++编写成为python扩展的内容写在空项目里。源文件中，需要写出C++下求取最大公因数的函数，后利用pybind11 的语法规则将其封装成MODULE。

在项目下添加setup.py以及pyproject.toml，此处注意需要保持模块的命名与源文件中pybind11封装的内容名一致。

在保存上述内容后，生成解决方案。

在项目中python环境-python包（PYPI），搜索框中输入pyproject.toml的路径（从属性中可以获得，但注意要删去最后的pyproject.toml），回车键通过pip安装

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132936211.png" alt="image-20220606132936211" style="zoom:50%;" />

安装成功，对所有python项目即可用，此时查看pip list

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606132950318.png" alt="image-20220606132950318" style="zoom:50%;" />

可以看到已经成功安装（此处的superfastcode2）

此处只需正常调用python模块，同时写出python计算最大公因数的函数，以COUNT为计算次数，在输出结果的时候，同时输出时间代价。

<img src="C:\Users\Dang_Yufan\AppData\Roaming\Typora\typora-user-images\image-20220606133003591.png" alt="image-20220606133003591" style="zoom:50%;" />

可以看到使用C++编写的扩展，相较于直接调用python，节省了20倍有余的时间代价。

### 三、总结

Cython与pybind11 为python编写C++的扩展模块，都通过消除解释器所需的时间开销，对效率有较明显的提升。具体实现上，代码的书写并不困难。环境配置上有较多需要注意的细节，但对配置其后的原理，有时尚还不求甚解，期望后续的学习可以解决在环境配置中“为什么”的疑惑。

> 参考文献：

> Microsoft docx : creating a c++extension for python
>
> https://docs.microsoft.com/en-us/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2019
>
> Python/C API Referenc
>
> https://docs.python.org/3/c-api/
>
> Extending python with C/C++
>
> [Extending Python with C or C++ — Python 3.10.4 documentation](https://docs.python.org/3/extending/extending.html)
>
> Pybind11简介
>
> [pybind11简介 - 非法关键字 - 博客园 (cnblogs.com)](https://www.cnblogs.com/linxmouse/p/9105494.html)

 分别采用pybind11 与 Cython实现相关功能，实现小作业最大公因数与最小公倍数，目录如下：