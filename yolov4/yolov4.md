## YOLOv4  

1.用单卡就能完成检测训练过程.  
2.Bag-of-Freebies and Bag-of-Specials
Bag-of-Freebies:指目标检测器在不增加推理损耗的情况下达到更好的精度，这些方法称为只需转变训练策略或只增加训练量成本。也就是说数据增扩、类标签平滑(Class label smoothing)、Focal Loss等这些不用改变网络结构的方法 
Bag-of-Special:用最新最先进的方法（网络模块）来魔改检测模型--插入模块是用来增强某些属性的，显著提高目标检测的准确性。比如SE模块等注意力机制模块，还有FPN等模块。  
3.除了在模型上进行魔改，作者还加了其他的技巧   
backbone主要是提取特征，去掉head也可以做分类任务  
Neck主要是对特征进行融合，这里有很多技巧在  
### 核心目标  
加快神经网络的运行速度，在生产系统中优化并行计算，而不是低计算量理论指标（BFLOP） 
作者实验对比了CSPResNext50、CSPDarknet53和EfficientNet-B3。从理论与实验角度表明：CSPDarkNet53更适合作为检测模型的Backbone  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-26.png)  
#### 总结:YOLOv4模型 = CSPDarkNet53 + SPP + PANet(path-aggregation neck) + YOLOv3-head  
### BoF和BoS的选择  
对于改进CNN通常使用一下方法:   
1.activation:ReLU, leaky-ReLU, (parametric-ReLU and SELU训练难度大,ReLU6专门量化网络的设计,没选用这3个激活函数), Swish,or **Mish**  
   
2.bbox regression loss:MSE, IoU->GIoU->DIoU->CIoU    

3.data augumentation:CutOut, **MixUp**, **CutMix**  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-36.png)
4.regulation:DropOut, DropPath, Spatial DropOut ,or **DropBlock**  
Dropout:整体随便扔    
Spatial Dropout: 按通道随机扔    
**DropBlock**:每个特征图按spatial块随机扔    
DropConnect:只在连接处随意扔，神经元不扔    

5.normalization:Batch Normalization (BN) , Cross-GPU Batch Normalization (CGBN or SyncBN) , Filter Response Normalization (FRN),or Cross-Iteration Batch Normalization (CBN)  

6.skip-connections:Residual connections, Weighted residual connections(from EfficientDet), Multi-input weighted residual connections, or Cross stage partial connections (CSP)  

对于以上方法，作者又进行了额外更改:  
**输入端的创新:**    
1.data augument:Mosaic,SAT(自我对抗训练)  
Mosaic优点:  
a. 丰富数据库：随机使用4张图片，随机缩放，再随机分布进行拼接，大大丰富了检测数据集，特别是随机缩放增加了很多小目标，让网络的鲁棒性更好  
b. 减少GPU：可能会有人说，随机缩放，普通的数据增强也可以做，但作者考虑到很多人可能只有一个GPU。  
因此Mosaic增强训练时，可以直接计算4张图片的数据，使得Mini-batch大小并不需要很大，一个GPU就可以达到比较好的效果  

2.Class label smoothing(标签平滑):通常，将bbox的正确分类表示为类别[0,0,0,1,0,0，...]的独热编码，并基于该表示来计算损失函数。但是，当模型对预测值接近1.0,变得过分确定时，通常是错误的，overfit，并且以某种方式忽略了其他预测值的复杂性。按照这种直觉，在某种程度上重视这种不确定性对类标签进行编码是更合理的。当然，作者选择0.9，因此[0,0,0,0.9，0 ....]代表正确的类

3.SAT:通过转换输入图片来反应了漏洞.首先，图片通过正常的训练步骤，然后用对于模型来说根据最有害的loss值来修改图片，而不是反向传播更新权重，在后面的训练中，模型不得不强制面对最困难的例子并学习它。 
 
4.BN-->CBN-->**CmBN**  
CBN:  
(1)作者认为连续几次训练iteration中模型参数的变化是平滑的
(2)作者将前几次iteration的BN参数保存起来，当前iteration的BN参数由当前batch数据求出的BN参数和保存的前几次的BN参数共同推算得出(Cross-Interation BN)
(3)训练前期BN参数记忆长度短一些，后期训练稳定了可以保存更长时间的BN参数来参与推算，效果更好  
CmBN:  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-48.png)  

## yolov4框架  
CSPDarknet53借鉴CSPNet,CSPNet全称是Cross Stage Paritial Network，主要从网络结构设计的角度解决推理中从计算量很大的问题。CSPNet的作者认为推理计算过高的问题是由于网络优化中的梯度信息重复导致的  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-50-54.png)  
yolov4创新点:   
（1）输入端：这里指的创新主要是训练时对输入端的改进，主要包括**Mosaic数据增强、cmBN、SAT自对抗训练**。
（2）BackBone主干网络：将各种新的方式结合起来，包括：**CSPDarknet53、Mish激活函数、Dropblock**
（3）Neck：目标检测网络在BackBone和最后的输出层之间往往会插入一些层，比如Yolov4中的SPP模块、FPN+PAN结构
（4）Prediction：输出层的锚框机制和Yolov3相同，主要改进的是训练时的损失函数CIOU_Loss，以及预测框筛选的nms变为DIOU_nms  
### backbone主干网络创新   
**CSPDarknet网络结构**    
优点:  
增加CNN学习能力，使得在轻量化的同时保持准确性,降低计算瓶颈,降低内存成本  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-02.png)   
每个CSP模块前面的卷积核的大小都是3×3，因此可以起到下采样的作用。  
因为Backbone有5个CSP模块，输入图像是608\*608，所以特征图变化的规律是：608->304->152->76->38->19,经过5次CSP模块后得到19\*19大小的特征图。  
而且作者只在Backbone中采用了Mish激活函数，网络后面仍然采用Leaky_relu激活函数。  
backbone卷积层个数:  
每个CSPX中包含3+2×X个卷积层，因此整个主干网络Backbone中一共包含2+（3+2×1）+2+（3+2×2）+2+（3+2×8）+2+（3+2×8）+2+（3+2×4）+1=72  
**DropBlock**  
**Mish**  

### Neck创新  
**SPP模块**  
采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合。   
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-09.png)   
**FPN+PAN**  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-15.png)  

原本的PANet网络的PAN结构中，两个特征图结合是采用shortcut操作，而Yolov4中则采用concat（route）操作，特征图融合后的尺寸发生了变化  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-51-22.png)  

不过这里需要注意几点：
注意一：
Yolov3的FPN层输出的三个大小不一的特征图①②③直接进行预测
但Yolov4的FPN层，只使用最后的一个76×76特征图①，而经过两次PAN结构，输出预测的特征图②和③。
这里的不同也体现在cfg文件中，这一点有很多同学之前不太明白。
比如Yolov3.cfg中输入时608×608，最后的三个Yolo层中，
第一个Yolo层是最小的特征图19×19，mask=6,7,8，对应最大的anchor box  
第二个Yolo层是中等的特征图38×38，mask=3,4,5，对应中等的anchor box  
第三个Yolo层是最大的特征图76×76，mask=0,1,2，对应最小的anchor box  
而Yolov4.cfg则恰恰相反  
第一个Yolo层是最大的特征图76×76，mask=0,1,2，对应最小的anchor box  
第二个Yolo层是中等的特征图38×38，mask=3,4,5，对应中等的anchor box  
第三个Yolo层是最小的特征图19×19，mask=6,7,8，对应最大的anchor box   
其他基础操作：
1. Concat：张量拼接，维度会扩充，和Yolov3中的解释一样，对应于cfg文件中的route操作。
2. Add：张量相加，不会扩充维度，对应于cfg文件中的shortcut操作  
