---
title: "Memory considerations"
teaching: 10
exercises: 10
questions:
- "Why do I need to send my data to the GPU in batches?"
- "How can I monitor the GPU memory usage?"
objectives:
- "Understand the issues around GPU memory limits."
- "Provide information on where to learn more about GPU memory management."
keypoints:
- "GPU memory is not the only consideration when setting the batch size."
- "Memory limits will depend on both allocated and reserved memory."
---

<iframe width="560" height="315" src="https://www.youtube.com/embed/tE5oMcA_tX4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


When it comes to memory usage, there are two main things to consider: the size of your training data and the size of your model. Remember that all the learnable parameters in your model require space in memory and that parameters where historic gradients are being calculated and used also *accumulate* in memory. In this lesson we will look at the memory being used by the training data and the memory being used by the model itself.

# Mini-batching

There are two reasons that we sub-divide the data into mini-batches during training:

* To produce a better loss curve;
* To make sure that the data fit into GPU memory.

 &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp;  &nbsp; &nbsp;&nbsp; &nbsp; *"Friends don't let friends use mini-batches larger than 32"* - Yann LeCunn

Most deep learning applications use [stochastic gradient algorithms](https://hsf-training.github.io/hsf-training-ml-webpage/02-mltechnical/index.html) to update the values of the weights and biases in the network during training. Rather than computing these updates on a sample by sample basis, they instead use parameter updates based on gradient averages over small subsets of the full training set. These are known as *mini-batches* and the *batch size* specifies how many samples are in each mini-batch.

Exactly how to set the batch size for a particular machine learning application is a multi-faceted problem. On one hand, increasing the batch size can make more efficient use of the parallel processes on the GPU, but on the other hand large batch sizes have been shown to adversely affect the convergence of the training and cause models to over-fit the training data. Typically the batch size is treated as a hyper-parameter than can be tuned using the validation data set.

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
> Take the final network from you run time tests (`hidden_size=2000`) and try using the whole training data set as a single batch. What happens?
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


# Monitoring memory usage

You may have noticed in the output from the challenges above that the error message associated with running out of memory typically includes some information like this:

~~~
RuntimeError: CUDA out of memory. Tried to allocate 8.74 GiB (GPU 0; 15.90 GiB total capacity; 8.84 GiB already allocated; 6.32 GiB free; 8.86 GiB reserved in total by PyTorch)
~~~
{: .language-python}

The allocation of memory on a GPU is not super simple. As well as the memory that is used to store tensor data, software applications will also typically *reserve* additional memory in a cache in order to speed up processing that requires access to memory. The way that the amount of reserved memory is decided depends on the software library itself.

In PyTorch it is possible to monitor the allocated memory for a particular GPU using:

~~~
a = torch.cuda.memory_allocated(0)
~~~
{: .language-python}

and to monitor the cached memory using:

~~~
c = torch.cuda.memory_reserved(0)
~~~
{: .language-python}

### Memory usage in neural networks

The dataset we're using to train the model in this example is pretty small in terms of volume, so small changes to a reasonable batch size (16, 32, 64 etc.) will not have a huge effect on the GPU memory usage in this case. However, we are using a fully-connected neural network which contains a large number of learnable parameters.

The learnable parameters in a fully-connected layer - `nn.Linear(m, n)` in PyTorch - use O(nm) memory: that is to say, the memory requirements scale *quadratically* with the number of features.

> ## Challenge
> By adding additional layers, work out how deep you can make your network before running out of GPU memory when using a batch size of 32.
>
> > ## Solution
> >
> > The solution to this challenge will depend on exactly what kind of GPU you're running on. However, typically it will happen for a network with 5 hidden layers, each containing 5000 neurons.
> {: .solution}
{: .challenge}


# Where can I get help if I have memory problems on the GPU?

For PyTorch there are some [helpful FAQs](https://pytorch.org/docs/stable/notes/faq.html) available, which outline common coding issues that can cause memory not to be released efficiently. There is also a useful description of the [routines available to monitor memory usage](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).
