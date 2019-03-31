# PCVch03续

记录学习Python Computer Vision的过程

第四次

## 全景图原理

**全景图**：在同一位置（即图像的照相机位置相同）拍摄的两幅或者多幅图像是单应性相关的。我们经常使用这些约束将很多图像缝补起来，拼成一个大的图像来创建全景图像。

那么全景拼图可以包括三大部分：特征点提取与匹配、图像配准、图像融合

### 特征点提取和匹配

在进行图像缝补之前我们需要对图像进行特征点匹配，这里使用SIFT特征自动找到匹配对应。SIFT是具有很强的稳健性的描述子，能够比其他描述子，产生更少的错误的匹配。SIFT特征的原理，理论资料我已经在上一篇博客说过了，这里就不再提及。下面附上实现SIFT特征匹配代码：

```python
import sift

# set paths to data folder
featname = ['C:/Users/ZQQ/Desktop/advanced/study/computervision/images/ch03/shu0'+str(i+1)+'.sift' for i in range(2)] 
imname = ['C:/Users/ZQQ/Desktop/advanced/study/computervision/images/ch03/shu0'+str(i+1)+'.jpg' for i in range(2)]

# extract features and match
l = {}
d = {}
for i in range(2): 
    sift.process_image(imname[i],featname[i])
    l[i],d[i] = sift.read_features_from_file(featname[i])

matches = {}
for i in range(1):
    matches[i] = sift.match(d[i+1],d[i])

```

### 图像配准

在对两图像做完特征匹配之后，要对图像进行配准。图像配准是一种确定待拼接图像间的重叠区域以及重叠位置的技术，它是整个图像拼接的核心。本节采用的是基于特征点的图像配准方法，即通过匹配点对构建图像序列之间的变换矩阵，从而完成全景图像的拼接。

变换矩阵H求解是图像配准的核心，其求解的算法流程如下。

-  检测每幅图像中特征点
- 计算特征点之间的匹配
- 计算图像间变换矩阵的初始值。
- 迭代精炼H变换矩阵。
- 引导匹配。用估计的H去定义对极线附近的搜索区域，进一步确定特征点的对应。
- 重复迭代4）和5）直到对应点的数目稳定为止。

设图像序列之间的变换为投影变换

可用4组最佳匹配计算出H矩阵的8 个自由度参数hi=( i=0,1,...,7)，并以此作为初始值。

为了提高图像配准的精度，本节采用RANSAC算法对图像变换矩阵进行求解与精炼，达到了较好的图像拼接效果。RANSAC算法的基本原理可通过图9-1来描述。

![image](https://github.com/zengqq1997/PCVch03-/blob/master/ransac.jpg)

RANSAC算法的思想简单而巧妙：首先随机地选择两个点，这两个点确定了一条直线，并且称在这条直线的一定范围内的点为这条直线的支撑。这样的随机选择重复数次，然后，具有最大支撑集的直线被确认为是样本点集的拟合。在拟合的误差距离范围内的点被认为是内点，它们构成一致集，反之则为外点。根据算法描述，可以很快判断，如果只有少量外点，那么随机选取的包含外点的初始点集确定的直线不会获得很大的支撑，值得注意的是，过大比例的外点将导致RANSAC算法失败。在直线拟合的例子中，由点集确定直线至少需要两个点；而对于透视变换，这样的最小集合需要有4个点。

在模型确定以及最大迭代次数允许的情况下，RANSAC总是能找到最优解。经过我的实验，对于包含80%误差的数据集，RANSAC的效果远优于直接的最小二乘法。

图9-1中蓝色点属于内点（正确点），而第红色点属于外点（偏移点）。此时用最小二乘法拟合这组数据，实际的最佳拟合直线是那条穿越了最多点数的蓝色实线。

RANSAC在此次图像拼接的代码实现部分如下：

```python
# function to convert the matches to hom. points
def convert_points(j):
    ndx = matches[j].nonzero()[0]
    fp = homography.make_homog(l[j+1][ndx,:2].T) 
    ndx2 = [int(matches[j][i]) for i in ndx]
    tp = homography.make_homog(l[j][ndx2,:2].T) 
    
    # switch x and y - TODO this should move elsewhere
    fp = vstack([fp[1],fp[0],fp[2]])
    tp = vstack([tp[1],tp[0],tp[2]])
    return fp,tp


# estimate the homographies
model = homography.RansacModel() 

#fp,tp = convert_points(0)
#H_12 = homography.H_from_ransac(fp,tp,model)[0] #im 1 to 2 

fp,tp = convert_points(0)
H_01 = homography.H_from_ransac(fp,tp,model)[0] #im 0 to 1 

# tp,fp = convert_points(2) #NB: reverse order
# H_32 = homography.H_from_ransac(fp,tp,model)[0] #im 3 to 2 

# tp,fp = convert_points(3) #NB: reverse order
# H_43 = homography.H_from_ransac(fp,tp,model)[0] #im 4 to 3    
```



### 拼接图像

估计出图像间的单应性矩阵（上面使用的RANSAC算法），现在我们需要将所有的图像扭曲到一个公共的图像平面上。通常，这里的公共平面为中心图像平面（否则，需要进行大量的变性）。根据图像间变换矩阵H，可以对相应图像进行变换以确定图像间的重叠区域，并将待融和图像映射到到一幅新的空白图像中形成拼接图。需要注意的是，由于普通的相机在拍摄照片时会自动选取曝光参数，这会使输入图像间存在亮度差异，导致拼接后的图像缝合线两端出现明显的明暗变化。因此，在融和过程中需要对缝合线进行处理。进行图像拼接缝合线处理的方法有很多种，例如快速简单的加权平滑算法处理拼接缝问题。本次实验采用的是创建一个很大的图像，如将图像中全部填充0，使其和中心图像平行，然后将所有的图像扭曲到上面。由于我们这次的实验图像都是用照相机水平旋转拍摄的，因此我们可以使用一个叫简单的步骤：将中心图像左边或者右边的区域填充0，以便为扭曲的图像腾出空间。下面是实现代码：

```python
def panorama(H,fromim,toim,padding=2400,delta=2400):
    """ Create horizontal panorama by blending two images 
        using a homography H (preferably estimated using RANSAC).
        The result is an image with the same height as toim. 'padding' 
        specifies number of fill pixels and 'delta' additional translation. """ 
    
    # check if images are grayscale or color
    is_color = len(fromim.shape) == 3
    
    # homography transformation for geometric_transform()
    def transf(p):
        p2 = dot(H,[p[0],p[1],1])
        return (p2[0]/p2[2],p2[1]/p2[2])
    
    if H[1,2]<0: # fromim is to the right
        print 'warp - right'
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim,zeros((toim.shape[0],padding,3))))
            fromim_t = zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                                        transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim,zeros((toim.shape[0],padding))))
            fromim_t = ndimage.geometric_transform(fromim,transf,
                                    (toim.shape[0],toim.shape[1]+padding)) 
    else:
        print 'warp - left'
        # add translation to compensate for padding to the left
        H_delta = array([[1,0,0],[0,1,-delta],[0,0,1]])
        H = dot(H,H_delta)
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0],padding,3)),toim))
            fromim_t = zeros((toim.shape[0],toim.shape[1]+padding,toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                                            transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0],padding)),toim))
            fromim_t = ndimage.geometric_transform(fromim,
                                    transf,(toim.shape[0],toim.shape[1]+padding))
    
    # blend and return (put fromim above toim)
    if is_color:
        # all non black pixels
        alpha = ((fromim_t[:,:,0] * fromim_t[:,:,1] * fromim_t[:,:,2] ) > 0)
        for col in range(3):
            toim_t[:,:,col] = fromim_t[:,:,col]*alpha + toim_t[:,:,col]*(1-alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t*alpha + toim_t*(1-alpha)
    
    return toim_t
```

现在在图像中使用该操作，函数如下：

```python
# warp the images
delta = 2000 # for padding and translation

im1 = array(Image.open(imname[0]), "uint8")
im2 = array(Image.open(imname[1]), "uint8")
im_01 = warp.panorama(H_01,im1,im2,delta,delta)

im1 = array(Image.open(imname[0]), "f")
im_02 = warp.panorama(dot(H_12,H_01),im1,im_12,delta,delta)

im1 = array(Image.open(imname[3]), "f")
im_32 = warp.panorama(H_32,im1,im_02,delta,delta)

im1 = array(Image.open(imname[4]), "f")
im_42 = warp.panorama(dot(H_32,H_43),im1,im_32,delta,2*delta)
```

## 实例

此次实验我们在集美大学拍摄了三组照片分别有

- 室内场景，我们拍摄学生宿舍。
- 室外景深落差较大的场景，我们拍摄了嘉庚图书馆
- 室外景深落差较小的场景，我们拍摄了中山纪念馆

### 室内场景实验

#### 原图

![image](https://github.com/zengqq1997/PCVch03-/blob/master/su01.jpg)

![image](https://github.com/zengqq1997/PCVch03-/blob/master/su02.jpg)

#### 结果图

![image](https://github.com/zengqq1997/PCVch03-/blob/master/suresult.jpg)

![image](https://github.com/zengqq1997/PCVch03-/blob/master/suresult1.jpg)

### 室外景深落差大实验

#### 原图

![image](https://github.com/zengqq1997/PCVch03-/blob/master/shu02.jpg)

![image](https://github.com/zengqq1997/PCVch03-/blob/master/shu03.jpg)

#### 结果图

![image](https://github.com/zengqq1997/PCVch03-/blob/master/shuresult.jpg)

![image](https://github.com/zengqq1997/PCVch03-/blob/master/shuresult1.jpg)

### 室外景深落差小实验

#### 原图

![image](https://github.com/zengqq1997/PCVch03-/blob/master/ji01.jpg)

![image](https://github.com/zengqq1997/PCVch03-/blob/master/ji02.jpg)

#### 结果图

![image](https://github.com/zengqq1997/PCVch03-/blob/master/jiresult.jpg)

![image](https://github.com/zengqq1997/PCVch03-/blob/master/jiresult1.jpg)

正如你所看到的，拍摄的角度不同，图像的曝光度不同，在单个图像的边界上存在边缘效应。在商业化的创建全景图像软件里有额外的操作来对强度进行归一化。

## 小结

本次实验实现了多张图像的全景拼接

