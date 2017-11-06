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
A couple of months ago, I tried to find a generic approach to monitor the data quality of ten of thousands of files containing trading data running through our systems everyday. To try to get alerted ahead of our clients, especially systemetic incidents that widely impact all customers on the market such as data lost due to system network outage. 

I decided to use Multivariate Distribution, a.k.a. Anomaly Dection in Machine Learning. It may be one of simplest Machine Learning model out there. It is just a simple statistical model. The beauty of this model is that it can 'dynamically' adapt to the characteristic of the data and it works with arbitary number of features. 

### 2 The Model
The core model is very simple:
* Build a multivariate distribution from historical data which captures the characteristic of every features as well as relationship between features
* Generated the probability for an event based on the distribution. 

Anomaly is identified by the model returning a probability below a predefined threshold.

... To Be Finished ...

#### 2.1 Data Preprocessing
... To Be Finished ...

#### 2.2 Calculation
... To Be Finished ...

### 3 Other Thoughts
... To Be Finished ...


### 4. Misc
Jupyter Notebook on [Github](https://github.com/edwinwzhe/blog_projects/blob/master/anomaly_detection/AnomalyDetection.ipynb)

### 5. Reference
[1] TBA
