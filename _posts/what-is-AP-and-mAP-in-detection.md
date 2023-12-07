---
title: what is AP and mAP in detection
date: 2023-07-21 12:25:43
index_img: /img/ap_figure.png
tags: [mAP]
author : NA-Wen
category: MLdetails
---

AP and mAP are basic measurement for the percision of detection .Here is a detailed introduction. 
<!-- more -->
### some low level definition
TP-true and positive: correct detection 
FP-false and positive : incorrect detection
TN-true and negative : This is the background region correctly not detected by the model. This metric is not used in object detection because such regions are not explicitly annotated when preparing the annotations.
FN-false and negative : missed ground truth 

Then comes to : how to define a detection result TP/FP/TN/FN?
we use IoU
### IoU
$$ IoU=\frac{gt\cap pd}{gt \cup pd}$$
which evaluates the overlap between gt and pd.
By setting a threshold $\alpha$ on IoU ,we can determine a result 's type.
TP: the IoU is larger than $\alpha$
FP: the IoU is smaller than $\alpha$
FN:  the IoU is smaller than $\alpha$(meaning miss ground truth)
### P and R
Percision :
evaluate the degree of exactness of the model in identifying only relevant objects
$$ P=\frac{TP}{all \:detections(TP+FP)}$$
Recall :
the ability of the model to detect all ground truths— proposition of TPs among all ground truths
$$R=\frac{TP}{all \: ground \: truth(TP+FN) }$$

The precision-recall (PR) curve is a plot of precision and recall at varying confidence values. 
1. when $\alpha$ is bigger ,  P raises but R decreases
2. when $\alpha$ is smaller , R raises but P decreases

### AP 
AP@α is Area Under the Precision-Recall Curve(AUC-PR) evaluated at α IoU threshold.
$$ AP @ \alpha= \int_0^1 p(r)d r$$ 
$\alpha$ is the threshold .If you see metrics like AP50 and A75, they mean AP calculated at IoU=0.5 and IoU=0.75, respectively.

A bigger area under P-R curve represents a higher P and R , but usually ,the curve is not monotonically decreasing but zig-zag-like plot .We will use some interpolaton method to remove this property .

1. 11-point interpolation method
   $$AP@_{\alpha 11}=\frac{1}{11}\sum_{i=0}^{i=10}p_i ,p_i =p(r=\frac{i}{10}) $$ 
   the interpolated precision at recall value, r — It is the highest precision for any recall value r'≥ r
2. All — point interpolation method
   $$AP@_{\alpha}=\sum_i(r_{i+1}-r_{i})p_i $$








mAP is the average AP for each class.
>AP can be calculated over a range of thresholds. Microsoft COCO calculated the AP of a given category/class at 10 different IoU ranging from 50% to 95% at 5% step-size, usually denoted AP@[.50:.5:.95]. Mask R-CNN reports the average of AP@[.50:.5:.95] simply as AP. 
### Example 
Here is an example to make all things above clear:
They contain 12 detection (red boxes) and 9 ground truths (green). Each detection has a class marked by a letter and the model confidence. In this example, consider that all the detections are of the same object class, and the IoU threshold is set α = 50 per cent. IoU values are shown in Table 1 below.
![](/img/ap_figure.png)
note: if multi detection for only one object , we only select the one having the highest confidence , to justify whether it is TP .
![](/img/ap_result.png)
**the calculation pipeline**: find each detection box type ->calculate P and R -> draw P-R curve under each threshold $\alpha$-> calculate AP -> average AP for each class to get mAP

### References:
1. https://towardsdatascience.com/on-object-detection-metrics-with-worked-example-216f173ed31e
