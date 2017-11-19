---
title:  "Anomaly Dection"
date:   2017-09-08 15:04:23
categories: [python, machine learning]
tags: [python, machine learning]
---
## A Generic Approach to System/Data Monitoring

<br/>

#### _Disclaimer_
_The original data I played with are private. The sample data used in this post is fake but capturing the real relative changes so that the result is identical to the result of real data_


### 1 Intro
A couple of months ago, I tried to find a generic approach to monitor the data quality of ten of thousands of files containing trading data running through our systems everyday. To try to get alerted ahead of our clients, especially systematic incidents that widely impact all customers on the market such as data lost due to system/network outage. 

I decided to use Multivariate Distribution, a.k.a. Anomaly Detection in Machine Learning. It may be one of simplest Machine Learning model out there. It is just a simple statistical model. The beauty of this model is that it can 'dynamically' adapt to the characteristic of the data and it works with arbitary number of features. 

### 2 The Model
The core model is very simple:
* Build a multivariate distribution from historical data which captures the characteristic of every features as well as relationship between features
* Generated the probability for an event based on the distribution. 

Anomaly is identified by the model returning a probability below a predefined threshold.

__Considered a scenario__ involves only two features cpu usage and network bandwidth usage of a web server. 
Figure 1 shows the bivariate distribution of the historical data. It shows positive correlation between two features. It make sense to me the CPU is busy when the network has more traffic.

![Multivariate Distribution]({{ site.url }}/assets/anomaly_detection/multivariate_distribution.png)
_Figure 1: Bivariate Distribution_

The ellipse in green is the threshold. Dots fall outside of the ellipse is considered anomaly. For example, the CPU is very busy when the network has few traffic. 

The strength of this model is that it does not only capture the probability distribution of individual features but also captures the relationship between features. However, it is also the limitation of the model. It cannot be used on data that fluctuates severely. The model will stay silent on any incident.

### 3 The Data
#### 2.1 Data Preprocessing
... To Be Finished ..................................

#### 2.2 Calculation
... To Be Finished ...

### 3 Other Thoughts
... To Be Finished ...


### 4. Misc
Jupyter Notebook on [Github](https://github.com/edwinwzhe/machine_learning/blob/master/anomaly_detection/AnomalyDetection.ipynb)

### 5. Reference
[1] TBA
