### Prediction 创新  

现有目标检测的Loss普遍采用预测bbox与ground truth bbox的1-范数，2-范数来作为loss。但是evaluation的时候却又采用IOU来判断是否检测到目标。显然二者有一个Gap，即loss低不代表IOU就一定小。例如:  
![](https://github.com/weiweia92/pictures/blob/master/IOU/Screenshot%20from%202020-07-04%2012-23-57.png)
问题1：即状态1的情况，当预测框和目标框不想交时，IOU=0，无法反应两个框距离的远近，此时损失函数不可导，`IOU_Loss`无法优化两个框不相交的情况  
问题2：即状态2和状态3的情况，当两个预测框大小相同，两个IOU也相同，IOU_Loss无法区分两者相交情况的不同  

因此2019年出现了GIOU来进行改进  

### GIOU  
![](https://github.com/weiweia92/pictures/blob/master/IOU/Screenshot%20from%202020-07-04%2012-24-13.png)
GIOU特点:  
* GIOU是IOU的下界，且取值范围为(-1, 1]。当两个框不重合时，IOU始终为0，不论A、B相隔多远，但是对于GIOU来说，A，B不重合度越高（离的越远），GIOU越趋近于-1  
* 与IoU类似，GIoU也可以作为一个距离，loss可以用L(GIOU)=1-GIOU来计算,GIoU对物体的大小不敏感 
* GIOU收敛速度慢，一般不能很好的收敛SOTA算法　　
问题：状态1、2、3都是预测框在目标框内部且预测框大小一致的情况，这时预测框和目标框的差集都是相同的，因此这三种状态的GIOU值也都是相同的，这时GIOU退化成了IOU，无法区分相对位置关系。  
基于这个问题，2020年的AAAI又提出了DIOU_Loss    

### DIOU  

Complete IoU Loss
bbox回归的三个要素：overlap area, central point distance and aspect ratio。IOU loss 和 GIoU loss可以看作是只考虑了overlap area，DIOU loss 考虑了overlap area,和central point distance 。而 CIoU将 aspect ratio也考虑了进来.  
  
### CIOU   
![](https://github.com/weiweia92/pictures/blob/master/IOU/Screenshot%20from%202020-07-04%2012-24-23.png)
收敛速度比DIOU更快。　　　　
![](https://github.com/weiweia92/pictures/blob/master/IOU/Screenshot%20from%202020-07-04%2012-24-36.png)
