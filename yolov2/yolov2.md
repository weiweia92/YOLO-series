## YOLOv2/YOLO9000  
在yolov1的基础上进行改进　　

## YOLOv2     
### 1.BN替代Dropout   
神经网络学习过程本质就是为了学习数据分布,一旦训练数据与测试数据的分布不同,那么网络的泛化能力也大大降低;另外一方面，一旦每批训练数据的分布各不相同(batch 梯度下降),那么网络就要在每次迭代都去学习适应不同的分布,这样将会大大降低网络的训练速度。 对数据进行预处理（统一格式、均衡化、去噪等）能够大大提高训练速度，提升训练效果。

解决办法之一是对数据都要做一个归一化预处理。YOLOv2网络通过在每一个卷积层后添加batch normalization，极大的改善了收敛速度同时减少了对其regularization方法的依赖（舍弃了dropout优化后依然没有过拟合）
Batch Normalization和Dropout均有正则化的作用。但是Batch Normalization具有提升模型优化的作用，这点是Dropout不具备的。所以BN更适合用于数据量比较大的场景。  

### 2.High Resolution Classifier  
预训练分类模型采用了更高分辨率的图片  
YOLOv1先在ImageNet（224x224）分类数据集上预训练模型的主体部分（大部分目标检测算法），获得较好的分类效果，然后再训练网络的时候将网络的输入从224x224增加为448x448。但是直接切换分辨率，检测模型可能难以快速适应高分辨率。所以YOLOv2增加了在ImageNet数据集上使用448x448的输入来finetune分类网络这一中间过程（10 epochs），这可以使得模型在检测数据集上finetune之前已经适用高分辨率输入。使用高分辨率分类器后，YOLOv2的mAP提升了约4%。

### 3.Darknet-19网络框架　　
提出了新的分类网络darknet-19作为基础模型　　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-32.png)  
用3\*3的filter来提取特征，1\*1的filter减少output channels.  
最后的卷积层用3\*3的卷积层代替,filter=1024,然后用1\*1的卷积层将7\*7\*1024-->7\*7\*125 注:125 = (5\*(4+1+20)    
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-40.png)   
　　
使用1\*1的卷积是为了减少参数,Darknet-19进行了5次降采样，但是在最后一层卷积并没有添加池化层，目的是为了获得更高分辨率的Feature Map,在3\*3卷积中间添加了1\*1卷积，Feature Map之间的一层非线性变化提升了模型的表现能力 

### 4.Convolutional with anchor boxes  
YOLO采用全连接层来直接预测bounding boxes,yolov2去除了YOLO的全连接层，采用先验框（anchor boxes）来预测bounding boxes。    
首先，去除了一个pooling层来提高卷积层输出分辨率。然后，修改网络输入尺寸：考虑到很多情况下待检测物体的中心点容易出现在图像的中央，将448×448改为416，使特征图只有一个中心。物品（特别是大的物品）更有可能出现在图像中心。　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-47.png)  
使用416\*416经过5次降采样之后生成的Feature Map的尺寸是13\*13 ，这种奇数尺寸的Feature Map获得的中心点的特征向量更准确。其实这也和YOLOv1产生7\*7的理念是相同的；采用anchor boxes，提升了精确度。YOLO每张图片预测98个boxes，但是采用anchor boxes，每张图片可以预测超过1000个boxes

### 5.Dimension Clusters  
为了确定对训练数据具有最佳覆盖范围的前K个bbox，我们在训练数据上运行K-均值聚类，以找到前K个聚类的质心。由于我们是在处理边界框而不是点，因此无法使用规则的空间距离来测量数据点距离。所以我们使用IOU.聚类的目的是anchor boxes和临近的ground truth有更大的IOU值，这和anchor box的尺寸没有直接关系。自定义的距离度量公式：d(box,centroid)=1-IOU(box,centroid),距离越小，IOU值越大。  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-19-55.png)  
图中看出anchor个数选为5时，结果最佳。右侧表示anchor的形状  

### 6.Direct Location Prediciton　　
对anchor的偏移量进行预测，但是如果不限制我们的预测将再次随机化，pre-bbox很容易向任何方向偏移。因此，每个位置预测的边界框可以落在图片任何位置，这导致模型的不稳定性，在训练时需要很长时间来预测出正确的offsets.yolov2预测了5个参数(tx,ty,tw,th,to),应用sigma函数来限制偏移范围。　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-20-04.png)　　
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-20-11.png)

### 7.Fine-Grained Features
5次maxpooling得到的13×13大小的feature map对检测大物体是足够了，但是对于小物体还需要更精细的特征图（Fine-Grained Features）.YOLOv2提出了一种passthrough层来利用更精细的特征图。YOLOv2所利用的Fine-Grained Features是26×26大小的特征图（最后一个maxpooling层的输入），对于Darknet-19模型来说就是大小为26×26×512的特征图。passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上。前面的特征图维度是后面的特征图的2倍，passthrough层抽取前面层的每个2×2的局部区域，然后将其转化为channel维度，对于26×26×512的特征图，经passthrough层处理之后就变成了13×13×2048的新特征图（特征图大小降低4倍，而channles增加4倍)，这样就可以与后面的13×13×1024特征图连接在一起形成13×13×3072的特征图，然后在此特征图基础上卷积做预测。  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-06-05%2014-20-18.png)  

### 8.Training   
分类任务训练  

作者采用ImageNet1000类数据集来训练分类模型。训练过程中，采用了 random crops, rotations, and hue, saturation, and exposure shifts等data augmentation方法。预训练后，作者采用高分辨率图像（448×448）对模型进行finetune。

检测任务训练  

作者将分类模型的最后一层卷积层去除，替换为三层卷积层（3×3,1024 filters），最后一层为1×1卷积层，filters数目为需要检测的数目。对于VOC数据集，我们需要预测5个boxes，每个boxes包含5个适应度值，每个boxes预测20类别。因此，输出为125（5*20+5*5） filters。最后还加入了passthough 层。  

## YOLO9000  
尽管基于深度学习的计算机视觉的最新进步极大地改善了对象检测应用程序，但是对象检测的范围仍然局限于一小部分对象，这是由于用于检测的标记数据集数量有限。

最流行的检测数据集Pascal VOC可检测20个类别，类似地，MS COCO可检测80个类别，其中包含数千到数百个图像，而分类数据集则包含数百万个具有数十万个类别的图像。为了缩小检测和分类任务之间的数据集大小差距，本文介绍了YOLO9000，这是一种实时对象检测系统，可检测9000多个对象类别，YOLO9000利用海量的分类数据集，使用联合训练算法，正确地定位了未标记的对象，以进行检测。 COCO检测数据集和ImageNet分类数据集。  
