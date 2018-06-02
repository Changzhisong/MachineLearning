
机器学习  
=======  
  
****  
### Author:Song  
### E-mail:Z.S.Chang@qq.com  
****  
本系列是我学习机器学习过程中的一些总结。主要参考了李航《统计学习方法》、周志华的《机器学习》及一些技术博客。时的一些总结。如果有数学公式无法显示，推荐使用Google Chrome与GitHub with MathJax插件组合浏览！！！  
## 目录  
* [机器学习的基本术语](#机器学习的基本术语 )
* [没有免费午餐定理(NFL)](#没有免费午餐定理（NFL）)
* [模型评估与选择](#模型评估与选择)
	* [评估方法](#评估方法)
	* [调参与最终模型](#调参与最终模型)
	* [性能度量](#性能度量)
* [感知机学习算法](#感知机学习算法)  
	* [为什么叫“感知机"](#为什么叫“感知机")
	* [对偶形式](#对偶形式)
* [k近邻法](#k近邻法)  
	* [kd树](#kd树实现)
	* [球树（BallTree）](#球树（BallTree）实现) 
* [决策树](#决策树)  
	* [决策树学习基本算法](#决策树学习基本算法)
	* [最优划分属性方法](#最优划分属性方法)
	* [ID3](#ID3)
	* [C4.5](#C4.5)
	* [CART](#CART)
	* [剪枝](#剪枝)
	* [sklearn中决策树参数](#sklearn中决策树参数)
* [支持向量机SVM](#SVM)  
	* [函数间隔与几何间隔](#1.几个概念)
	* [线性可分SVM](#2.线性可分SVM（硬间隔）)
	* [线性SVM](#3.线性SVM（软间隔）)
	* [非线性SVM](#4.非线性SVM)
	* [SVR](#5.SVR)
	* [SMO（序列最小最优化）算法](#6.SMO（序列最小最优化）算法)
	* [sklearn中SVM参数](#7.sklearn中SVM参数)
* [朴素贝叶斯](#朴素贝叶斯)  
	* 最大似然估计
	* 贝叶斯估计 
	 
* 
***  
---
--- 
### 机器学习的基本术语   
* **样本：**   
数据集中每条记录是关于一个事件或对象的描述，称为一个示例（instance）或者一个样本（sample）。但有时候整个数据集也称作一个样本，可根据上下文判断具体指哪个。

* **属性/特征：**   
反映事件或对象在某方面的表现或性质的事项，称为属性（attribute）或特征（feature）。

* **属性空间/样本空间/输入空间：**   
属性张成的空间称为属性空间（attribute space）样本空间（sample space）、输入空间。例如：属性有n个，可以将n个属性张成一个n维空间，每个对象都可以在这个空间中找到自己的坐标点，即空间中的每一个点对应一个坐标向量，因此也把一个示例称为一个特征向量（feature vector）。  

* **学习/训练：**   
从数据中学得模型的过程（learning/training） 

* **训练数据：**   
训练过程中使用的数据（training set）  

* **训练样本/训练示例/训练例：**   
训练数据中的每个样本称为一个训练样本（training sample）

* **训练集：**   
训练样本组成的集合（training set）

* **假设及真相：**  
学得模型对应了关于数据的某种潜在的规律，称为假设（hypothesis）；这种潜在规律自身称为真相或真实（ground-truth），学习过程就是为了找出或者逼近真相。

* **标记：**   
示例具有的结果信息，称为标记（label）

* **样例：**   
拥有了标记信息的示例，称为样例（example），但有时也称为样本。一般的用(xi, yi)表示第i个样例， 是示例xi的标记，Y为标记空间。

* **标记空间/输出空间：**  
所有标记的集合（label space）。

* **分类：**   
预测的是离散值，则此类学习任务称为分类（classification）

* **回归：**  
 预测的是连续值，此类学习任务称为回归（regression）

* **二分类：**   
只涉及两个类别的分类（binary classification），通常称其中一个类为正类（positive class），另一个类别为反类或负类（negative class）。

* **多分类：**  
涉及多个类别的分类。  
`一般地，预测任务是希望通过对训练集{(x1, y1), (x2 , y2),…, (xm, ym)}进行学习，建立一个输入空间X到输出空间Y的映射f：X→Y。对二分类任务，通常令Y={-1，+1}或{0, 1}；对于多分类任务，|Y|>2；对于回归任务，Y=R，R为实数集。`

* **测试：**   
学得模型后。使用其进行预测的过程称为测试（testing）

* **测试样本：**   
被预测的样本称为测试样本（testing sample）。例如在学得f后，对测试例x，可得到其预测标记y = f(x)

* **簇：**   
对对象做聚类（clustering），即将训练集中的对象分成若干组，每组称为一个簇。

* **监督学习/无监督学习：**   
根据训练数据是否拥有标记信息学习任务可大致划分为两大类：监督学习（supervised learning）和无监督学习（unsupervised learning），分类和回归是前者的代表，为聚类是后者的代表。

* **误差**  
学习器的实际预测输出与样本的真实输出之间的差异

* **训练误差或经验误差（empirical error）**  
学习器在训练集上的误差

* **泛化误差（generalization error）**  
学习器在在新样本上的误差

* **泛化能力：**   
学得模型适用于新样本的能力

----
---


###  没有免费午餐定理（NFL）  
你在所有问题出现的机会相同、或所有问题同等重要的前提下，没有一种算法比随机胡猜的效果好。换句话说，假设所有数据的分布可能性相等，当我们用任一分类做法来预测未观测到的新数据时，对于误分的预期是相同的。简而言之，NFL的定律指明，如果我们对要解决的问题一无所知且并假设其分布完全随机且平等，那么任何算法的预期性能都是相似的。  
`NFL定理的寓意是让我们清楚地认识到，脱离具体问题，空泛的谈论什么学习算法更好是毫无意义的，因为若考虑所有潜在的问题，则所有学习算法都一样好。要谈论算法的相对优劣，必须要针对具体的学习问题。`
**学习算法自身的归纳偏好与问题是否匹配，往往会起到决定性的作用。**  




----
---

### 模型评估与选择  

为了得到泛化误差小的学习器，应该从训练样本中尽可能学到适用于所有潜在样本的普遍规律，这样才能在遇到新样本是做出正确的判别。     
**过拟合（overfitting）** 通常是由于学习能力过于强大以至于把训练样本所包含的不太一般的特性都学到了，而 **欠拟合** 则通常是由于学习能力低而造成的。过拟合是无法彻底避免的，我们所做的是缓解尽量减小其风险。    

*  **评估方法**   
通常，采用测试集来测试学习器对新样本的判别能力，然后以测试集上的测试误差（testing error）来近似泛化误差。测试集的样本也是从样本真实分布中独立同分布采样而得，测试集尽可能与训练集互斥，即测试样本尽量不在训练集中出现。  
划分训练集和测试集有以下几种方法。  

	* **留出法(hold-out)**   
直接将数据集D划分为两个互斥的集合，其中一个集合作为训练集S，另一个作为测试节T，在S上训练出模型后，用T来评估其测试误差，作为对泛化误差的估计。   
`注意：`  
（1）训练、测试集的划分尽可能保持数据分布的一致性，比如测试集和训练集的正负样本比例一致。  
（2）单次使用留出法得到的估计往往不够稳定可靠，一般要采用多次随机划分、重复进行实验评估后取平均值作为留出法的评估结果。  
（3）一般测试集小时，评估结果的方差较大，训练集小时，评估结果的偏差较大。通常将2/3~4/5的样本用于训练，剩余的用于测试。  

	* **交叉验证法(cross validation)**   
先将数据集D划分为k个大小相似的互斥子集Di，每个子集都尽可能保持数据分布的一致性，即从D中通过分层采样得到。然后，每次用k-1个子集的并集作为训练集，余下的那个子集作为测试集，这样就可以获得k组训练/测试集，进行k次训练和测试，最终返回这k次测试结果的均值。通常又称为**“k-折交叉验证”**(k-fold cross validation)，k常取值10,5,20。    
`注意：`  
与留出法类似，k折交叉验证通常需要使用不同的划分重复多次，最后的评估结果是这几次k折结果的均值。例如常见的:10次10折交叉验证。  

	* **特例(留一法（Leave-One-Out）**)   
	假定D中包含m个样本，若令k=m，对其进行k折交叉验证。留一法不受随机样本划分的方式影响，即不需向重复实验。留一法的评估结果往往被认为比较准确，但也不是绝对的，而且其开销较大。  

	* **自助法(bootstrapping)**    
以自助采样（也叫可重复采样或又放回采样）为基础。给定包含m个样本的数据集D，重复有放回的采样m次后得到一个包含m个样本的数据集D’，显然，D中有一部分样本会多次在D’中出现，也有一部分从未出现。    
样本在m次采样中始终未采到的概率为 .取极限得到：$$ \lim \limits_{m\rightarrow \infty}({1- \frac {1}{m})^m=\frac{1}{e}=0.368}$$  
即约36.8%的样本未出现在D’中，于是将D’用作训练集（会有重复），D\D’用作测试集（\表示集合减法）。这样实际评估的模型与期望评估的模型都使用m个训练样本。对于这样得到的测试结果也称为“包外估计”（out-of-bag estimate）  
`注意：`  
（1）自助法在数据集较小、难以有效划分训练/测试集时很有用。   
（2）自助法产生的数据集改变了初始数据集的分布，这会引入估计偏差，因此，在初始数据量足够时，留出法和交叉验证更常用。   
* **调参与最终模型**  
学习算法的参数配置不同，学得模型的性能也会不同，因此，除了要对适用学习算法进行选择，还需对算法参数进行设定，这就是“调参”。  
在模型选择完成后，学习算法和参数配置已设定，此时应该用数据集D重新训练模型，这个模型在训练过程中使用了所以m个样本，这才是我们最终提交给用户的模型。  
模型实际使用中遇到的数据称为测试数据，为了加以区分，模型评估与选择中用于评估测试的数据集称为验证集（validation set），在研究不同算法的泛化能力时，我们用测试集上的判别效果来估计模型在实际使用时的泛化能力，而把训练数据另外划分为训练集和验证集，基于验证集上的性能来进行模型选择和调参。    

* **性能度量**
	* 均方误差和均方根误差  
	
	* 错误率与精度  
		* 错误率（error rate）：分类错误的样本数/样本总数  
		* 精度（accuracy）：(1-错误率)*100%  
	* 查准率(准确率)、查全率(召回率)与F1  
	
		* 查准率（准确率P）：真的正/预测正   
		* 查全率（召回率R）：真的正/原来正  
		* F1度量：P、R的调和平均  
		* $F_\beta$:用$\beta$来调节P、R哪个影响更大  
		
	* PR曲线  
	以查准率P为纵轴、查全率R为横轴作图.一般P-R曲线的线下面积越大其性能越好  

	* ROC与AUC  
		* ROC曲线：以真正例率（TPR：真的正/预测正 P）为纵轴，假正例率（FPR:实际负例中，错误预测率。假的正/原来负）为横轴  
		* AUC:ROC线下面积。  

---  
___  


### 感知机学习算法  
* **原始形式**  
  
|**输入**| 训练数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$ ，其中$x_i$∈$R^n$,$y_i$∈{$\{+1,-1\}$}，学习率$η(0<η≤1)$ |  
|:---------:|:-------------|  
|**输出**| w，b，感知机模型$f(x)=sign(w·x+b)$ |  
|**步骤** |（1）选取初始值w0，b0；<br>（2）在训练集中选取数据$(x_i,y_i)$；<br>（3）如果存在误分类点，即$y_{i}(w·x+b)≤0$，则<br>　　　　　　　　　$w=w+ηy_ix_i$<br>　　　　　　　　　$b=b+ηy_i$ <br>（4）转（2），直至训练集中没有误分类点。 |  
|**解释** | （1）目标是求参数w，b，使得损失函数：$min_{w,b}$ $L(w,b)=- \sum_{x_{i}∈M} y_{i}(w·x_i+b)$极小化。其中M为误分类点的集合。<br>（2）采用随机梯度下降法（SGD）求解，即极小化过程中不是一次使M中所有误分类点的梯度下降，而是一次随机选取一个误分类点使其梯度下降。随机梯度下降的效率要高于批量梯度下降（batch gradient descent）<br>（3）当一个实例点被误分类，即位于分离超平面的错误一侧时，则调整w，b的值，使分离超平面向该误分类点的一侧移动，以减少该误分类点与超平面简的距离，直至超平面越过该误分类点使其被正确分类。|  
|**优缺点**|（1）在感知机模型中，只要给出的数据是线性可分的，那么我们就能证明在有限次迭代过程中，能够收敛；<br>（2）算法着重考虑是否将所有训练数据分类正确，而不考虑分的有多好，即存在无穷多个解，其解由于初值或不同的迭代顺序而可能有所不同；<br>（3）感知机主要的本质缺陷是它不能处理线性不可分问题；<br>（4）感知机因为是线性模型，所以不能表示复杂的函数，如异或，即感知机是无法处理异或问题，因为训练集线性不可分。|  
  
`为什么叫“感知机”：`感知机是生物神经细胞的简单抽象。神经细胞结构大致可分为：树突、突触、细胞体及轴突。单个神经细胞可被视为一种只有两种状态的机器——激动时为‘是’，而未激动时为‘否’。神经细胞的状态取决于从其它的神经细胞收到的输入信号量，及突触的强度（抑制或加强）。当信号量总和超过了某个阈值时，细胞体就会激动，产生电脉冲。电脉冲沿着轴突并通过突触传递到其它神经元。为了模拟神经细胞行为，与之对应的感知机基础概念被提出，如权量（突触）、偏置（阈值）及激活函数（细胞体）。  

* **对偶形式**  

|**输入** |训练数据集 $T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$，其中$x_i$∈$R^n$,$y_i$∈{$\{+1,-1\}$}，学习率$η(0<η≤1)$|  
|:---------:|:-------------|  
|**输出** |a，b，感知机模型$f(x)=sign(\sum_{j=1}^{N}a_jy_jx_j·x+b)$ ,其中$a=(a_i,a_2,...,a_N)$|  
|**步骤**|（1）选取初始值a=0，b=0；<br>（2）在训练集中选取数据$(x_i,y_i)$；<br> （3）如果存在误分类点，即$y_{i}(\sum_{j=1}^{N}a_jy_jx_j·x+b)≤0$，则<br>　　　　　　　　　　$a=a_i+η$<br>　　　　　　　　　　$b=b+ηy_i$ <br>（4）转（2），直至训练集中没有误分类点。|  
|**解释**|（1）为了方便，可以预先将训练集中实例间的内积计算出来并以矩阵的形式存储，即Gram矩阵。$G_{N×N}=[x_{i}·y_{j}]$，可加快计算速度。<br> （2）由 $w=w+ηy_ix_i$，$b=b+ηy_i$， 可知，对于误分类点$(x_i, y_i)$,w, b关于该点的增量分别为$a_iy_ix_i$，$a_iy_i$，其中，$n_i$表示第i个实例点由于误分而进行更新的次数。因此最终w, b可表示为：<br>　　　　　　　　　　$w=\sum_{i=1}^{N}a_iy_ix_i$ <br>　　　　　　　　　　$b=\sum_{i=1}^{N}a_iy_i$ |  

----  
---  


### k近邻法  
**输入：** 训练数据集T，实例x，参数k  
  
**输出：** 实例x所属的类y  
  
**步骤：**  
  
（1）根据给定的距离度量，在训练集中T中找出与$x$最邻近的$k$个点，涵盖着$k$个点为$x$的邻域  
（2）在领域中根据分类决策规则（如多数表决）决定$x$的类别$y$  
  
**解释：**  
  
（1）给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例，这k个实例的多数属于某个类，就把该输入实例分为这个类。  
（2）分类决策规则通常采用多数表决，多数表决等价于经验风险最小化。  
（3）通常采用**交叉验证法**来选取最优的k值，k值太小，泛化误差偏大，k值取太大，泛化误差减小。  
（4）回归输出的是k个最近邻点取得**平均值**或其他。  
（5）对异常点不敏感  
  
**实现:**  

* 暴力实现  
* **kd树实现**  
（1）构建kd树  
	* 不断的用垂直于坐标轴的超平面将k维空间切分（一般选择方差最大者的坐标轴），构成一系列的k维超矩形区域。  
	* 划分点选择该坐标轴上的中位数为划分点，划分成左右子树。  
	* **终止条件**：直到子区域没有实例点  
	
	（2）用kd树的最近邻搜索  
	* 给定一个目标节点，根据该点各坐标轴的数据找到包含目标节点的叶节点（一个超矩形区域），以该目标节点为圆心，到该叶节点的样本实例点（每次划分是以一个样本的实例点的某个维度为切分点的，切分线经过该点）的距离为半径，得到一个超球体，最近的节点一定在该球体内部或表面（即该叶节点），然后从该节点出发依次回退到父节点，不断的查找另一边的子节点区域中是否有与超球体相交的点，更新与目标点**最近**的节点。  
	* **终止条件**：直到回退到根节点，此时的最近点记为所求。  
	* **将找到的最近点标记为已选，重复上面的步骤，忽略已选点，找齐k个点**。  
	* 分类：投票法；回归：取均值   
	* 建立kd-tree的时间复杂度为O(k*n*logn)   
  
* **球树（BallTree）实现**   
	* **改进动机**：  
	kd树把k维空间划分成一个一个超矩形，但是矩形区域的**角**很难处理（因为判断是否在其他子节点区域）。超球体很好的解决了这个问题。  
	* 步骤和kd树类似（略）    

----  
---  


### 决策树  
* **决策树学习基本算法**  
  
|**输入**| 训练数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$ ,特征集A|  
|:----:|:-----|  
|**输出**| 决策树T 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |  
|**过程**|　　函数 TreeGenerate(D, A) <br>1　　生成结点node; <br>2　　**if** D中样本全属于同一类别C **then** <br>3　　　　将node标记为C类叶结点; **return** <br>4　　**end if** <br>5　　**if** A == ∅ **OR** D中样本在A上取值相同 **then** <br>6　　　　 将node标记为叶结点，其类别标记为D中样本数最多的类; **return** <br>7　　**end if** <br>8　　从A中选择**最优划分属性**$a_\*$; <br>9　　**for** $a_\*$ 的每一个值 $a_\*^v$ **do** <br>10　　　　为node生成一个分支;令$D_v$表示D中在$a_\*$上取值为$a_\*^v$的样本子集; <br>11　　　　**if** $D_v$ 为空 **then** <br>12　　　　　　将分支结点标记为叶结点，其类别标记为D(其父节点)中样本最多的类; **return** <br>13　　　　**else** <br>14　　　　　　以TreeGenerate($D_v$, A＼{$a_{\*}$})为分支结点 <br>15　　　　**end if** <br>16 　 **end for** <br>|  
  
* **最优划分属性方法**  
（1）信息增益（ID3算法）  
	* **熵：** 表示的随机变量的不确定性，
	* **信息增益：** 则是已知该特征的信息而使得数据的不确定性减少的程度。  
	* **缺点**： 采用信息增益准则会对可取值数目较多的特征有所偏好。因为对于取值数目较多的特征，更容易使得数据更“纯”，即各个分支节点中所包含的样本会更多的属于同一个类，也就是说，该特征会使不确定性（样本熵、信息熵、经验熵）减少的更多，比如极端情况：当对于某个特征，其对每个样本都有一个不同的取值，那么该特征是最大信息增益特征，因为若采用这个特征，其每个分支节点的纯度已达最大，然而，这样的决策树显然不具有泛化能力，无法对新样本进行有效预测。  
	
	（2）信息增益比（C4.5）  
	* 信息增益比准则对可取值数目较少的特征有所偏好，因此C4.5算法并不是直接选择信息增益率最大的候选划分特征，而是使用了一个启发式：先从候选划分特征中找出信息增益高于平均水平的属性。再从中选择信息增益率最大的。  
  
	（3） 基尼指数 （CART）  
	（4） 最小化均方误差（回归）  
* **剪枝**  
（1）剪枝是决策树对付“过拟合”的主要手段，基本策略有“预剪枝”和“后剪枝”；  

	（2）预剪枝是指在决策树生成过程中，对每个节点在划分前先进行估计，若当前节点的划分不能带来决策树泛化性能提升，则停止划分并将当前节点标记为叶节点。预剪枝不仅降低了过拟合的风险，还显著减少了决策树的训练时间开销和测试时间开销；  

	（3）后剪枝是先从训练集生成一颗完整的决策树，然后自底向上地对非叶子结点进行考察，若将该节点对应的子数替换为叶子节点能带来决策树的泛化性能提升，则将该子树替换为叶节点。  

* **sklearn 中决策树参数**
	* **class_weight：**  类别权重  
	指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重，或者用“balanced”，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的"None"

	* **max_leaf_nodes：** 最大叶子节点数

	* **criterion：** 特征选择标准  
	DecisionTreeClassifier：可以使用"gini"或者"entropy"，前者代表基尼系数，后者代表信息增益。一般使用默认的基尼系数"gini"就可以了，即CART算法  
	DecisionTreeRegressor：可以使用"mse"或者"mae"，前者是均方差，后者是和均值之差的绝对值之和。推荐使用默认的"mse"

	* **splitter：** 特征划分点选择标准  
	可以使用"best"或者"random"。前者在特征的所有划分点中找出最优的划分点。后者是随机的在部分划分点中找局部最优的划分点。

	* **max_features：** 划分时考虑的最大特征数
		
	* **max_depth：** 决策树最大深
	
	* **min_samples_split：** 内部节点再划分所需最小样本数
	
	* **min_samples_leaf：** 叶子节点最少样本数  
	
	* **min_impurity_split：** 节点划分最小不纯度  
	这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。
	* **presort：** 数据是否预排序
	
---  
---  

### 逻辑斯蒂回归  
* 
 
---  
---  

### SVM  
* **1.几个概念**    
	* **函数间隔$\gamma^{'}$：**
						$$\gamma^{'} = y(w^Tx + b)$$  
		* $|w^Tx + b|$表示点x到超平面的距离。通过观察$w^Tx + b$和y是否同号，判断分类是否正确。  
		* 对于训练集中m个样本点对应的m个函数间隔的最小值，就是整个训练集的函数间隔。函数间隔并不能正常反映点到超平面的距离，当分子成比例的增长时，分母也是成倍增长。
	* **几何间隔$\gamma$：**
				$$\gamma = \frac{y(w^Tx + b)}{||w||_2} =\frac{\gamma^{'}}{||w||_2}$$
		* 几何间隔才是点到超平面的真正距离。
	* **支持向量：**
		* 距离超平面最近的几个样本，满足$y_i(w^Tx_i + b)= 1$，则它们被称为支持向量。
		* 支持向量到超平面的距离为$1/||w||_2$,两个异类支持向量之间的距离为$2/||w||_2$。  
	![svm](https://images2015.cnblogs.com/blog/1042406/201611/1042406-20161124144326487-1331861308.jpg)　  

* **2. 线性可分SVM（硬间隔）**  

**输入：** 线性可分训练集${(x_1,y_1), (x_2,y_2), ..., (x_m,y_m),}$,其中x为n维特征向量。y为二元输出，值为1，或者-1。  
**输出：** 分离超平面的参数$w^{\*}$和$b^{\*}$和分类决策函数。   
**步骤：**   
（1）构造约束优化问题  
优化函数定义为：

$$max \ \frac{1}{||w||_2} \qquad s.t  \quad y_i(w^Tx_i + b) \geq 1 \ (i =1,2,...m)$$  

由于$\frac{1}{||w||_2}$的最大化等同于$\frac{1}{2}||w||_2^2$的最小化。这样SVM的优化函数（**原始问题**）等价于：

$$min \ \frac{1}{2}||w||_2^2  \qquad s.t \quad y_i(w^Tx_i + b)  \geq 1 \ (i =1,2,...m)$$  

通过**拉格朗日对偶性**将优化目标转化为无约束的优化函数：

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$L(w,b,\alpha)&space;=&space;\frac{1}{2}||w||_2^2&space;\sum\limits_{i=1}^{m}&space;{\alpha_i&space;{[y_i&space;(w^Tx_i&space;&plus;&space;b)&space;-&space;1]}}&space;\qquad&space;s.t.&space;\quad&space;\alpha_i&space;\geq&space;0$$" title="$$L(w,b,\alpha) = \frac{1}{2}||w||_2^2 \sum\limits_{i=1}^{m} {\alpha_i {[y_i (w^Tx_i + b) - 1]}} \qquad s.t \quad \alpha_i \geq 0$$" />
原始问题的对偶问题是极大极小问题：

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$\underbrace{max}_{\alpha_i&space;\geq&space;0}&space;\underbrace{min}_{w,b}&space;\&space;L(w,b,\alpha)$$" title="$$\underbrace{max}_{\alpha_i \geq 0} \underbrace{min}_{w,b} \ L(w,b,\alpha)$$" />

求$\underbrace{min}_{w,b} \ L(w,b,\alpha)$对$\alpha$的极大，即是**对偶问题**：

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$\underbrace{min}_{\alpha}&space;\frac{1}{2}&space;\sum\limits_{i=1}^{m}&space;\sum\limits_{j=1}^{m}&space;\alpha_i&space;\alpha_j&space;y_i&space;y_j&space;(x_i&space;\bullet&space;x_j)&space;-&space;\sum\limits_{i=1}^{m}&space;\alpha_i$$" title="$$\underbrace{min}_{\alpha} \frac{1}{2} \sum\limits_{i=1}^{m} \sum\limits_{j=1}^{m} \alpha_i \alpha_j y_i y_j (x_i \bullet x_j) - \sum\limits_{i=1}^{m} \alpha_i$$" /> 

$$ s.t. \ \sum\limits_{i=1}^{m}\alpha_iy_i = 0 $$   

$$ \alpha_i \geq 0  \qquad i=1,2,...m $$  	

（2）通过序列最小最优化算法（**SMO算法**）求对偶优化问题的$\alpha$向量的解$\alpha^{\*}$。  
（3）计算$w^{\*} = \sum\limits_{i=1}^{m}\alpha_i^{\*}y_ix_i$   
（4）找出所有的支持向量，假设有S个，即满足$\alpha_s > 0$对应的样本$(x_s,y_s)$，通过 $y_s(\sum\limits_{i=1}^{m}\alpha_iy_ix_i^Tx_s+b) = 1$，同时根据$y^2_s=1$，计算出每个支持向量$(x_x, y_s)$对应的$b_s^{\*}$，计算出这些$b_s^{\*} = y_s - \sum\limits_{i=1}^{m}\alpha_iy_ix_i^Tx_s$。所有的$b_s^{\*}$对应的平均值即为最终的$b^{\*} = \frac{1}{S}\sum\limits_{i=1}^{S}b_s^{\*}$   
（5）最终的分类超平面为：$w^{\*} \bullet x + b^{\*} = 0$，最终的分类决策函数为：$f(x) = sign(w^{\*} \bullet x + b^{\*})$    

**解释：**   
（1）数据集必须线性可分。  
（2）采用拉格朗日对偶性求解对偶问题的优点:  
	　　　	A. 对偶问题往往更容易求解  
　　　		B.	自然引入核函数，进而推广到非线性分类问题  
（3）**KKT条件**：  
　　　A.对参数（w、b）求导等于0；  
　　　B.约束条件；  
　　　C.拉格朗日乘子大于等于0；  
　　　D.对应的拉格朗日乘子乘以约束条件等于0。   
（4）训练完成后，大部分的训练样本都不需要保留，最终模型仅与支持向量有关。  

* **3. 线性SVM（软间隔）**  

**动机：** A.假如训练集中有一些异常点，将这些异常点去掉后，剩下的大部分都是线性可分的，此时不能使用硬间隔最大化来求超平面了。B.很难确定合适的核函数使线性可分。  
**输入：** 训练集${(x_1,y_1), (x_2,y_2), ..., (x_m,y_m),}$,其中x为n维特征向量。y为二元输出，值为1，或者-1。  
**输出：** 分离超平面的参数$w^{\*}$和$b^{\*}$和分类决策函数。  
**（1）** 选择一个惩罚系数$C>0$, 构造约束优化问题  
对训练集里面的每个样本$(x_i,y_i)$引入了一个松弛变量$\xi_i \geq 0$,使得：  
$$ y_i(w\bullet x_i +b) \geq 1- \xi_i $$  同时对每一个松弛变量$\xi_i$, 支付一个代价$\xi_i$，得到**软间隔原始问题**：  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;min\frac{1}{2}||w||_2^2&space;&plus;C\sum\limits_{i=1}^{m}\xi_i&space;$$" title="$$ min\frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i $$" />

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;s.t&space;\quad&space;y_i&space;(w^Tx_i&space;&plus;&space;b)&space;\geq&space;1&space;-&space;\xi_i&space;(i&space;=1,2,...m)&space;$$" title="$$ s.t \quad y_i (w^Tx_i + b) \geq 1 - \xi_i (i =1,2,...m) $$" />

$$ \xi_i \geq 0 \quad (i =1,2,...m)$$<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;\xi_i&space;\geq&space;0&space;\quad&space;(i&space;=1,2,...m)&space;$$" title="$$ \xi_i \geq 0 \quad (i =1,2,...m) $$" />

$$min\frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i$$   

$$s.t \quad y_i (w^Tx_i + b) \geq 1 - \xi_i (i =1,2,...m)$$   

$$\xi_i \geq 0 \quad (i =1,2,...m)$$   


这里,$C>0$为惩罚参数，为协调两者关系的正则化惩罚系数。  
将软间隔最大化的约束问题用拉格朗日函数转化为无约束问题如下：    
$$L(w,b,\xi,\alpha,\mu) = \frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i - \sum\limits_{i=1}^{m}\alpha_i [y_i (w^Tx_i + b) - 1 + \xi_i] - \sum\limits_{i=1}^{m}\mu_i\xi_i$$   

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$L(w,b,\xi,\alpha,\mu)&space;=&space;\frac{1}{2}||w||_2^2&space;&plus;C\sum\limits_{i=1}^{m}\xi_i&space;-&space;\sum\limits_{i=1}^{m}\alpha_i&space;[y_i&space;(w^Tx_i&space;&plus;&space;b)&space;-&space;1&space;&plus;&space;\xi_i]&space;-&space;\sum\limits_{i=1}^{m}\mu_i\xi_i$$" title="$$L(w,b,\xi,\alpha,\mu) = \frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i - \sum\limits_{i=1}^{m}\alpha_i [y_i (w^Tx_i + b) - 1 + \xi_i] - \sum\limits_{i=1}^{m}\mu_i\xi_i$$" />

其中 $\mu_i \geq 0, \alpha_i \geq 0$,均为拉格朗日系数。 对偶问题是拉格朗日函数的**极大极小问题**：  

$$\underbrace{ min }_{\alpha} \quad \frac{1}{2} \sum\limits_{i=1,j=1}^{m} \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum\limits_{i=1}^{m} \alpha_i$$    

$$s.t \quad \sum\limits_{i=1}^{m}\alpha_iy_i = 0 \qquad $$   

$$0 \leq \alpha_i \leq C$$    

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$\underbrace{&space;min&space;}_{\alpha}&space;\quad&space;\frac{1}{2}&space;\sum\limits_{i=1,j=1}^{m}&space;\alpha_i&space;\alpha_j&space;y_i&space;y_j&space;x_i^T&space;x_j&space;-&space;\sum\limits_{i=1}^{m}&space;\alpha_i$$" title="$$\underbrace{ min }_{\alpha} \quad \frac{1}{2} \sum\limits_{i=1,j=1}^{m} \alpha_i \alpha_j y_i y_j x_i^T x_j - \sum\limits_{i=1}^{m} \alpha_i$$" />

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$s.t&space;\quad&space;\sum\limits_{i=1}^{m}\alpha_iy_i&space;=&space;0&space;\qquad&space;$$" title="$$s.t \quad \sum\limits_{i=1}^{m}\alpha_iy_i = 0 \qquad $$" />

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$0&space;\leq&space;\alpha_i&space;\leq&space;C$$" title="$$0 \leq \alpha_i \leq C$$" />

和线性可分SVM相比，仅仅是多了一个约束条件$0 \leq \alpha_i \leq C$。  
**（2）** 通过序列最小最优化算法（**SMO算法**）求对偶优化问题的$\alpha$向量的解$\alpha^{\*}$。  
**（3）** 计算$w^{\*} = \sum\limits_{i=1}^{m}\alpha_i^{\*}y_ix_i$  
**（4）** 找出所有的支持向量，假设有S个，即满足$\alpha_s > 0$对应的样本$(x_s,y_s)$，通过 $y_s(\sum\limits_{i=1}^{m}\alpha_iy_ix_i^Tx_s+b) = 1$，同时根据$y^2_s=1$，计算出每个支持向量$(x_x, y_s)$对应的$b_s^{\*}$，计算出这些$b_s^{\*} = y_s - \sum\limits_{i=1}^{m}\alpha_iy_ix_i^Tx_s$。所有的$b_s^{\*}$对应的平均值即为最终的$b^{\*} = \frac{1}{S}\sum\limits_{i=1}^{S}b_s^{\*}$  
**（5）** 最终的分类超平面为：$w^{\*} \bullet x + b^{\*} = 0$，最终的分类决策函数为：$f(x) = sign(w^{\*} \bullet x + b^{\*})$   
**解释：**  
**（1）** 另一种解释如下：  
$$ \underbrace{min}_{w,b}\sum\limits_{i=1}^{m} [1-y_i (w \bullet x_i + b)]_{+} + \lambda ||w||_2^2 $$  

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;\underbrace{min}_{w,b}\sum\limits_{i=1}^{m}&space;[1-y_i&space;(w&space;\bullet&space;x_i&space;&plus;&space;b)]_{&plus;}&space;&plus;&space;\lambda&space;||w||_2^2&space;$$" title="$$ \underbrace{min}_{w,b}\sum\limits_{i=1}^{m} [1-y_i (w \bullet x_i + b)]_{+} + \lambda ||w||_2^2 $$" />

其中$L(y(w \bullet x + b)) = [1- y_i (w \bullet x + b) ]_{+}$称为合页损失函数(hinge loss function)，下标+表示为： $$$$[z]_{+} = z, \quad {z>0} ; \ 0, \quad {z\leq 0}$$ 

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$[z]_{&plus;}&space;=\left\{\begin{matrix}&space;z&space;\quad{z>0}\\&space;0\quad&space;{z\leq&space;0}\&space;\end{matrix}$$" title="$$[z]_{+} =\left\{\begin{matrix} z \quad{z>0}\\ 0\quad {z\leq 0}\ \end{matrix}$$" />  

也就是说，如果样本点$(x_i,y_i)$被正确分类，且函数间隔$y(w \bullet x + b)$大于1时，损失是0，否则损失是$1-y(w \bullet x + b)$,如下图中的绿线。我们在下图还可以看出其他各种模型损失和函数间隔的关系：对于0-1损失函数，如果正确分类，损失是0，误分类损失1， 如下图黑线，可见0-1损失函数是不可导的。对于感知机模型，感知机的损失函数是$[-y_i(w \bullet x + b)]_{+}$，这样当样本被正确分类时，损失是0，误分类时，损失是$-y_i(w \bullet x + b)$，如下图紫线。对于逻辑回归之类和最大熵模型对应的对数损失，损失函数是$log[1+exp(-y (w \bullet x + b))]$, 如下图红线所示。  
![合页损失函数](https://images2015.cnblogs.com/blog/1042406/201611/1042406-20161125140636518-992065349.png)

* **4. 非线性SVM**  

为了使样本满足线性可分，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。  

**核函数**  
假设$\phi$是一个从低维的输入空间$\chi$（欧式空间的子集或者离散集合）到高维的希尔伯特空间的$\mathcal{H}$映射。那么如果存在函数$K(x,z)$，对于所有的$x, z \in \chi$，都有：$K(x, z) = \phi(x) \bullet \phi(z)$，那么就称$K(x, z)$为核函数。**$K(x, z)$的计算是在低维特征空间来计算的**，避免了在高维维度空间计算内积的计算量。  

* **线性核函数**  
线性可分SVM我们可以和线性不可分SVM归为一类，区别仅仅在于线性可分SVM用的是线性核函数。  
$$K(x, z) = x \bullet z $$  

* **多项式核函数**  
$$K(x, z) =(\gamma x \bullet z + r)^d$$  

* **高斯核函数**  
高斯核函数（Gaussian Kernel），在SVM中也称为径向基核函数（Radial Basis Function,RBF），它是非线性分类SVM最主流的核函数。libsvm默认的核函数。  
$$K(x, z) = exp(-\gamma||x-z||^2)$$  

* **Sigmoid核函数**  
scikit-learn中描述的形式  
$$K(x, z) = tanh ( \gamma x \bullet z + r)$$  

**算法**  
利用核技巧，将线性SVM扩展得到非线性SVM，**只需将线性SVM对偶中的内积换成核函数**，$x_i \bullet x_j = K(x_i, x_j)$。其他步骤一样。  


* **5.SVR**  

**$\epsilon$-不敏感损失函数**  
对于回归模型，目标是让训练集中的每个点$(x_i,y_i)$,尽量拟合到一个线性模型$y_i = w \bullet \phi(x_i ) +b$。对于一般的回归模型，通常采用均方差作为损失函数,但是SVM采用的是$\epsilon$-不敏感损失函数。  

对于常量$\epsilon > 0$，对于某一个点$(x_i,y_i)$，如果$|y_i - w \bullet \phi(x_i ) -b| \leq \epsilon$，则损失为0，如果$|y_i - w \bullet \phi(x_i ) -b| > \epsilon$,则对应的损失为$|y_i - w \bullet \phi(x_i ) -b| - \epsilon$，这个均方差损失函数不同，如果是均方差，那么只要$y_i - w \bullet \phi(x_i ) -b \neq 0$，就会有损失。

![sur](https://github.com/Changzhisong/MachineLearning/blob/master/image/svm-support-vector-regression-tube-vs-square-error.png)

**SVR原始问题**  

$$min \frac{1}{2}||w||_2^2 \qquad s.t \quad |y_i - w \bullet \phi(x_i ) -b| \leq \epsilon \quad (i =1,2,...m)$$
对每个样本$(x_i,y_i)$加入两个松弛变量 $\xi_i^{\lor} \geq 0 , \xi_i^{\land} \geq 0$, 则SVR问题为：  

$$min \frac{1}{2}||w||_2^2 + C\sum\limits_{i=1}^{m}(\xi_i^{\lor}+ \xi_i^{\land})$$   

$$s.t \quad  -\epsilon - \xi_i^{\lor} \leq y_i - w \bullet \phi(x_i ) -b \leq\epsilon +\xi_i^{\land}$$   

$$\xi_i^{\lor} \geq 0, \quad \xi_i^{\land} \geq 0 \quad (i = 1,2,..., m)$$  

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$min&space;\frac{1}{2}||w||_2^2&space;&plus;&space;C\sum\limits_{i=1}^{m}(\xi_i^{\lor}&plus;&space;\xi_i^{\land})$$" title="$$min \frac{1}{2}||w||_2^2 + C\sum\limits_{i=1}^{m}(\xi_i^{\lor}+ \xi_i^{\land})$$" />  



<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$s.t&space;-\epsilon&space;-&space;\xi_i^{\lor}&space;\leq&space;y_i&space;-&space;w&space;\bullet&space;\phi(x_i&space;)&space;-b&space;\leq\epsilon&space;&plus;\xi_i^{\land}$$" title="$$s.t -\epsilon - \xi_i^{\lor} \leq y_i - w \bullet \phi(x_i ) -b \leq\epsilon +\xi_i^{\land}$$" />  



<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$\xi_i^{\lor}&space;\geq&space;0,&space;\quad&space;\xi_i^{\land}&space;\geq&space;0&space;\quad&space;(i&space;=&space;1,2,...,&space;m)$$" title="$$\xi_i^{\lor} \geq 0, \quad \xi_i^{\land} \geq 0 \quad (i = 1,2,..., m)$$" />  

**SVR对偶问题**   
拉格朗日函数对偶性得对偶问题：   

$$ \underbrace{ min}_{\alpha^{\lor}, \alpha^{\land}}\; \frac{1}{2}\sum\limits_{i=1,j=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor})(\alpha_j^{\land} -\alpha_j^{\lor})K_{ij} + \sum\limits_{i=1}^{m}(\epsilon-y_i)\alpha_i^{\land}+ (\epsilon+y_i)\alpha_i^{\lor} $$   

$$ s.t \quad \sum\limits_{i=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor}) = 0$$   

$$ 0 < \alpha_i^{\lor} < C \; (i =1,2,...m)$$   

$$ 0 <\alpha_i^{\land} <C \; (i =1,2,...m)$$  

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;\underbrace{&space;min}_{\alpha^{\lor},&space;\alpha^{\land}}\;&space;\frac{1}{2}\sum\limits_{i=1,j=1}^{m}(\alpha_i^{\land}&space;-&space;\alpha_i^{\lor})(\alpha_j^{\land}&space;-\alpha_j^{\lor})K_{ij}&space;&plus;&space;\sum\limits_{i=1}^{m}(\epsilon-y_i)\alpha_i^{\land}&plus;&space;(\epsilon&plus;y_i)\alpha_i^{\lor}&space;$$" title="$$ \underbrace{ min}_{\alpha^{\lor}, \alpha^{\land}}\; \frac{1}{2}\sum\limits_{i=1,j=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor})(\alpha_j^{\land} -\alpha_j^{\lor})K_{ij} + \sum\limits_{i=1}^{m}(\epsilon-y_i)\alpha_i^{\land}+ (\epsilon+y_i)\alpha_i^{\lor} $$" />

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;s.t&space;\quad&space;\sum\limits_{i=1}^{m}(\alpha_i^{\land}&space;-&space;\alpha_i^{\lor})&space;=&space;0$$" title="$$ s.t \quad \sum\limits_{i=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor}) = 0$$" />  

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;0&space;<&space;\alpha_i^{\lor}&space;<&space;C&space;\;&space;(i&space;=1,2,...m)$$" title="$$ 0 < \alpha_i^{\lor} < C \; (i =1,2,...m)$$" />  


<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;$$&space;0&space;<\alpha_i^{\land}&space;<C&space;\;&space;(i&space;=1,2,...m)$$" title="$$ 0 <\alpha_i^{\land} <C \; (i =1,2,...m)$$" />

对于这个目标函数，利用SMO算法来求出对应的$\alpha^{\lor}, \alpha^{\land}$，进而求出回归模型系数$w, b$。$w = \sum\limits_{i=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor})\phi(x_i)$，当$\beta_i =\alpha_i^{\land}-\alpha_i^{\lor} = 0$时，$w$不受这些在误差范围内的点的影响。对于在边界上或者在边界外的点，$\alpha_i^{\lor} \neq 0, \alpha_i^{\land} \neq 0$，此时$\beta_i \neq 0$。


* **6. SMO（序列最小最优化）算法**  

**原始问题**  
$$ \underbrace{ min }_{\alpha}  \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jK(x_i,x_j) - \sum\limits_{i=1}^{m}\alpha_i $$ 

$$ s.t \quad  \sum\limits_{i=1}^{m}\alpha_iy_i = 0 $$

$$0 \leq \alpha_i \leq C$$

 <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\underbrace{&space;min&space;}_{\alpha}&space;\frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jK(x_i,x_j)&space;-&space;\sum\limits_{i=1}^{m}\alpha_i&space;$$&space;\\&space;.\quad&space;\quad&space;s.t&space;\quad&space;\sum\limits_{i=1}^{m}\alpha_iy_i&space;=&space;0&space;\\&space;.&space;\&space;\qquad&space;\qquad0&space;\leq&space;\alpha_i&space;\leq&space;C">

每个变量 $\alpha_i$对应于一个样本点$(x_i,y_i)$，有m个变量组成的向量$\alpha$需要在目标函数极小化的时候求出。  
如果所有变量$\alpha_i$的解都满足最优化问题的KKT条件，那么这个最优化问题的解就得到了，因为KKT条件是该最优化问题的充分必要条件。  

**SMO基本思路：**   
 采用一种启发式的方法。每次只优化两个变量，将其他的变量都视为常数。由于$\sum\limits_{i=1}^{m}\alpha_iy_i = 0$.假如将$\alpha_3, \alpha_4, ..., \alpha_m$　固定，那么$\alpha_1, \alpha_2$之间的关系也确定了。这样SMO算法将一个复杂的优化算法转化为一个比较简单的两变量优化问题（该问题有闭式解，不用调用数值优化算法计算，快）。这个简单的两变量优化问题的解应该更接近原始问题的解，因为这会使得原始问题的目标函数值变小。利用这种方法可以使得整个算法的速度大大提高。子问题的两个变量，一个是选择违反KKT条件最严重的那一个，另一个由约束条件自动确定。重复不断的分解为子问题并求解，直到所有变量$\alpha_i$的解满足KKT条件，进而达到求解原始问的的目的。  

**子问题**   
定义$K_{ij} = \phi(x_i) \bullet \phi(x_j)$由于$\alpha_3, \alpha_4, ..., \alpha_m$都成了常量，所有的常量从目标函数去除，这样得到子优化问题：
$$\underbrace{ min }_{\alpha_1, \alpha_1} \frac{1}{2}K_{11}\alpha_1^2 + \frac{1}{2}K_{22}\alpha_2^2 +y_1y_2K_{12}\alpha_1 \alpha_2 -(\alpha_1 + \alpha_2)+y_1\alpha_1\sum\limits_{i=3}^{m}y_i\alpha_iK_{i1} + y_2\alpha_2\sum\limits_{i=3}^{m}y_i\alpha_iK_{i2}$$  

$$s.t \quad  \alpha_1y_1 + \alpha_2y_2 = -\sum\limits_{i=3}^{m}y_i\alpha_i = \varsigma$$   

$$0 \leq \alpha_i \leq C  i =1,2$$  


<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\small&space;\underbrace{&space;min&space;}_{\alpha_1,&space;\alpha_1}&space;\frac{1}{2}K_{11}\alpha_1^2&space;&plus;&space;\frac{1}{2}K_{22}\alpha_2^2&space;&plus;y_1y_2K_{12}\alpha_1&space;\alpha_2&space;-(\alpha_1&space;&plus;&space;\alpha_2)&plus;y_1\alpha_1\sum\limits_{i=3}^{m}y_i\alpha_iK_{i1}&space;&plus;&space;y_2\alpha_2\sum\limits_{i=3}^{m}y_i\alpha_iK_{i2}&space;\\&space;.\quad&space;\&space;s.t&space;\quad&space;\alpha_1y_1&space;&plus;&space;\alpha_2y_2&space;=&space;-\sum\limits_{i=3}^{m}y_i\alpha_i&space;=&space;\varsigma&space;\\&space;.\quad&space;\&space;\qquad&space;\&space;0&space;\leq&space;\alpha_i&space;\leq&space;C&space;\quad&space;i&space;=1,2">

**两个变量的选择**  

* 第一个变量的选择  
SMO算法称选择第一个变量为外层循环，这个变量需要选择在训练集中违反KKT条件最严重的样本点。对于每个样本点，要满足的KKT条件$$\alpha_{i} = 0  <=> y_ig(x_i)\geq1 $$ $$ 0 < \alpha_{i}<C <=> y_ig(x_i)=1 $$ $$\alpha_{i}=C <=>  y_ig(x_i)\leq 1$$首先选择违反$0 < \alpha_{i}< C <=> y_ig(x_i)=1$这个条件的点（即支持向量）。如果这些支持向量都满足KKT条件，再选择违反$\alpha_{i}= 0 <=> y_i g(x_i) \geq1$和 $\alpha_{i}=C <=> y_ig(x_i)\leq 1$的点来检测是否满足KKT条件。 

* 第二个变量的选择  
选择第二个变量为内层循环，假设在外层循环已经找到了$\alpha_1$, 第二个变量$\alpha_2$的选择标准是让$|E1-E2|$有足够大的变化（$E_i$为对$x_i$的预测值与真实输出$y_i$的差值）。由于$\alpha_1$定了的时候,$E_1$也确定了，所以要想$|E1-E2|$最大，只需要在$E_1$为正时，选择最小的$E_i$作为$E_2$， 在$E_1$为负时，选择最大的$E_i$作为$E_2$，可以将所有的$E_i$保存下来加快迭代。  
如果内存循环找到的点不能让目标函数有足够的下降， 可以采用遍历支持向量点来做$\alpha_2$,直到目标函数有足够的下降， 如果所有的支持向量做$\alpha_2$都不能让目标函数有足够的下降，可以跳出循环，重新选择$\alpha_1$　  


**7. sklearn中SVM参数**  

* sklearn.svm模块中  
* 分类：SVC， NuSVC，和LinearSVC  
* 回归:SVR， NuSVR，和LinearSVR  
 具体详情参见[这里](http://www.cnblogs.com/pinard/p/6117515.html "SVM调参")


----  
---  
### 朴素贝叶斯法  

![朴素贝叶斯法](https://github.com/Changzhisong/MachineLearning/blob/master/image/朴素贝叶斯.jpg)  

---
