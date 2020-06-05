## YOLOv1  
大致流程:  
1.Resize成448\*448，图片分割得到7\*7网格(cell)  
2.CNN提取特征和预测：卷积部分负责提特征。全连接部分负责预测：a) 7\*7\*2=98个bounding box(bbox) 的坐标(cx,cy,w,h)和是否有物体的conﬁdence 。 b) 7\*7=49个cell所属20个物体的概率。  
3.过滤bbox（通过nms）  
 
网络最后的输出是 S×S×30 的数据块，yolov1是含有全连接层的，这个数据块可以通过reshape得到。也就是说，输出其实已经丢失了位置信息（在v2，v3中用全卷积网络，每个输出点都能有各自对应的感受野范围）。yolov1根据每张图像的目标label，编码出一个 S×S×(B\*5+20) 的数据块，然后让卷积网络去拟合这个target
### 1.S\*S 框
如果Ground Truth的中心落在某个单元（cell）内，则该单元负责该物体的检测  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-42-06.png)
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-42-47.png)  
我们将输出层Os\*s\*(c+B\*5) 看做一个三维矩阵，如果物体的中心落在第 (i,j) 个单元内，那么网络只优化一个 C+B\*5维的向量，即向量 O[i,j,:] 。 S 是一个超参数，在源码中 S=7,B是每个单元预测的bounding box的数量，B的个数同样是一个超参数，YOLO使用多个bounding box是为了每个cell计算top-B个可能的预测结果，这样做虽然牺牲了一些时间，但却提升了模型的检测精度。  

每个bounding box要预测5个值：bounding box(cx,cy,w,h)以及置信度P,定义confidence为Pr(object)\*IOU(pred,truth)。bbox的形状是任意猜测的，这也是后续yolov2进行优化的一个点。置信度P表示bounding box中物体为待检测物体的概率以及bounding box对该物体的覆盖程度的乘积 Pr(Object) * IOU(pred, truth)。其中(cx,cy)是bounding box相对于每个cell中心的相对位置， (w,h)是物体相对于整幅图的尺寸,范围均为[0,1]。  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-43-13.png) 

同时，YOLO也预测检测物体为某一类C的条件概率：Pr(class(i)|object)  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-43-37.png)  

对于每一个单元，YOLO值计算一个分类概率，而与B的值无关。在测试时  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-43-52.png)  
### 2.输入层　　
YOLO作为一个统计检测算法，整幅图是直接输入网络的。因为检测需要更细粒度的图像特征，YOLO将图像Resize到了448\*448而不是物体分类中常用的224\*224的尺寸。需要注意的是YOLO并没有采用VGG中先将图像等比例缩放再裁剪的形式，而是直接将图片非等比例resize。所以YOLO的输出图片的尺寸并不是标准比例的。 
### 3.骨干架构：VGG,leaky-relu  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-44-32.png)  
然而现在的一些文章指出leaky ReLU并不是那么理想，现在尝试网络超参数时ReLU依旧是首选。　　
### 4.Loss function  
#### loss = classification loss + localization loss + confidence loss  
#### classification loss  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-44-53.png)
#### localization loss  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-00.png)  
对不同大小的bbox预测中，相比于大bbox预测偏一点，小box预测偏一点更不能忍受。而sum-square error loss中对同样的偏移loss是一样。 为了缓和这个问题，作者用了一个比较取巧的办法，就是将box的width和height取平方根代替原本的height和width。 如下图：small bbox的横轴值较小，发生偏移时，反应到y轴上的loss（下图绿色）比big box(下图红色)要大。
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-06.png)

为了更加强调边界框的准确性，我们设定lambda(coord)=5(default)  
#### confidence loss  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-12.png)  
许多bbox不包含任何物体，这造成了类别不平衡问题，eg:我们训练的模型检测到背景的情况会比物体的情况多的多，为了解决这个问题，我们设定lambda(noobj)=0.5(default)  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2010-45-21.png)  
### 5.后处理　　　
测试样本时，有些物体会被多个单元检测到，NMS用于解决这个问题。 

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2013-49-58.png) 
### YOLO优点：　　
1.YOLO检测物体非常快 　
 
2.不像其他物体检测系统使用了滑窗或region proposal，分类器只能得到图像的局部信息。YOLO在训练和测试时都能够看到一整张图像的信息，因此YOLO在检测物体时能很好的利用上下文信息，从而不容易在背景上预测出错误的物体信息。和Fast-R-CNN相比，YOLO的背景错误不到Fast-R-CNN的一半    
3.YOLO可以学到物体的泛化特征.  
  
4.当YOLO在自然图像上做训练，在艺术作品上做测试时，YOLO表现的性能比DPM、R-CNN等之前的物体检测系统要好很多。因为YOLO可以学习到高度泛化的特征，从而迁移到其他领域。    

### YOLO的缺点
1.其精确检测的能力比不上Fast R-CNN更不要提和其更早提出的Faster R-CNN了。  

2.由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding box作为物体检测输出，即每个格子最多只预测出一个物体。当物体占画面比例较小，如图像中包含畜群或鸟群时，每个格子包含多个物体，但却只能检测出其中一个。这是YOLO方法的一个缺陷。  

3.YOLO对小物体的检测，其为了提升速度而粗粒度划分单元而且每个单元的bounding box的功能过度重合导致模型的拟合能力有限，尤其是其很难覆盖到的小物体。YOLO检测小尺寸问题效果不好的另外一个原因是因为其只使用顶层的Feature Map，而顶层的Feature Map已经不会包含很多小尺寸的物体的特征了。

4.Faster R-CNN之后的算法均趋向于使用全卷积代替全连接层，但是YOLO依旧笨拙的使用了全连接不仅会使特征向量失去对于物体检测非常重要的位置信息，容易产生物体的定位错误，而且会产生大量的参数，影响算法的速度。  

 
