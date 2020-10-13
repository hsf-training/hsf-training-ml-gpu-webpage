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
- "GPU memory is not the only consideration when setting the batch size."
- "."
---

# Mini-batching

There are two reasons that we sub-divide the data into mini-batches during training:

* To produce a better loss curve;
* To make sure that the data fit into GPU memory.

### &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *Friends don't let friends use mini-batches larger than 32* - Yann LeCunn

Most deep learning applications use [stochastic gradient algorithms](https://hsf-training.github.io/hsf-training-ml-webpage/02-mltechnical/index.html) to update the values of the weights and biases in the network during training. Rather than computing these updates on a sample by sample basis, they instead use parameter updates based on gradient averages over small subsets of the full training set. These are known as *mini-batches* and the *batch size* specifies how many samples are in each mini-batch. 

Exactly how to set the batch size for a particular machine learning application is a multi-faceted problem. On one hand, increasing the batch size can make more efficient use of the parallel processes on the GPU, but on the other hand large batch sizes have been shown to adversely affect the convergence of the training and cause models to over-fit the training data. Typically the bacth size is treated as a hyper-parameter than can be tuned using the validation data set.

A further consideration when you are using a large model on a GPU is whether there is enough memory available to fit both the model parameters and the training data into GPU memory. This is of particular importance if the size of an individual training sample itself is large, as it might be for image based machine learning model applications or those with thousands of features. 

> ## Common Errors
> If you run into memory problems on a GPU, you will see an error that looks like this:
>
> ~~~
> RuntimeError: CUDA out of memory.
> ~~~
> {: .language-python}
{: .callout}

> ## Challenge
> Take the final network from you run time tests (`hidden_size=5000`) and try using the whole training data set as a single batch. What happens?
> 
> > ## Solution
> > 
> > The first step to this challenge is working out how many data samples are in the full training set:
> > ~~~
> > print(len(train_loader.dataset))
> > ~~~
> > {: .language-python}
> > You can then set this number as your batch size.
> > 
> > The solution to this challenge will depend on exactly what kind of GPU you're running on. But in most cases you will run out of memory and see the error described above.
> {: .solution}
{: .challenge}

> ## Challenge
> The dataset we're using to train the model in this example is pretty small in terms of volume, so small changes to a reasonable batch size (16, 32, 64 etc.) will not have a huge effect on the GPU memory usage in this case. However, we are using a fully-connected neural network which contains a large number of learnable parameters. By adding additional layers, work out how deep we can make this network before running out of GPU memory when using a batch size of 32.
> 
> > ## Solution
> > 
> > The solution to this challenge will depend on exactly what kind of GPU you're running on. However, typically it will happen for a network with 5 hidden layers, each containing 5000 neurons.
> {: .solution}
{: .challenge}
