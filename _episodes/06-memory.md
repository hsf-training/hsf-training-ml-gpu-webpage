---
title: "Memory considerations"
teaching: 10
exercises: 10
questions:
- "Why do I need to send my data to the GPU in batches?"
- "How can I monitor the GPU memory usage?"
objectives:
- "Prepare the dataset for machine learning."
- "Get excited for machine learning!"
keypoints:
- "One must properly format data before any machine learning takes place."
- "Data can be formatted using scikit-learn functionality; using it effectively may take time to master."
---

# Mini-batching

There are two reasons that we sub-divide the data into mini-batches during training:

* To produce a better loss curve;
* To make sure that the data fit into GPU memory.

> ## Wise Words
> ### &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *Friends don't let friends use mini-batches larger than 32* - Yann LeCunn
{: .callout}

Most deep learning applications use stochastic gradient algorithms to update the values of the weights and biases in the network during training. Rather than computing these updates on a sample by sample basis, they instead use parameter updates based on gradient averages over small subsets of the full training set. These are known as *mini-batches* and the *batch size* specifies how many samples are in each mini-batch. 

Exactly how to set the batch size for a particular machine learning application is a multi-faceted problem. On one hand, increasing the batch size can make more efficient use of the parallel processes on the GPU, but on the other hand large batch sizes have been shown to adversely affect the convergence of the training and cause models to over-fit the training data. Typically the bacth size is treated as a hyper-parameter than can be tuned using the validation data set.

A further consideration when you are using a large model on a GPU is whether there is enough memory available to fit both the model parameters and the training data into GPU memory. This is of particular importance if the size of an individual training sample itself is large, as it might be for image based machine learning model applications or those with thousands of features. 

> ## Common Errors
> If you run into memory problems on a GPU, you will see an error that looks like this:
>
> ~~~
>
> ~~~
> 
{: .callout}


