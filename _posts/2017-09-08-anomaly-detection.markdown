---
title:  "Anomaly Dection"
date:   2017-09-08 15:04:23
categories: [python, machine learning]
tags: [python, machine learning]
---
## A Generic Approach to System/Data Monitoring

A couple of months ago, I tried to find a generic approach to monitor the data quality of ten of thousands of binary files per day containing trading data running through our systems everyday. To try to get alerted ahead of our clients, especially systemetic incidents that have a wide impact such as system outage. 

I decided to use Multivariate Distribution, a.k.a. Anomaly Dection in Machine Learning. It may be one of simplest Machine Learning model out there. Before the buzz word Machine Learning, it is just a simple statistical model. The beauty of this model is that it can 'dynamically' adapt to the characteristic of the data as the anomaly dection is done via the building of a distribution from the historical data. Also it works with arbitary number of features.

The core model is very simple - build a multivariate distribution from historical data - generated a probability for an event based on the distribution. Anomaly is identified by having a probability below the preset threshold. 

... To Be Finished ...