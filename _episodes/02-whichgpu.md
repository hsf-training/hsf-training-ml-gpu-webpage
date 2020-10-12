---
title: "Is a GPU available?"
teaching: 20
exercises: 0
questions:
- "How do I find out if a GPU is available?"
- "How can I determine the specifications of the GPU?"
- "How do I select to use the GPU?"
objectives:
- "Discuss the role of data, models, and loss functions in machine learning."
- "Discuss the role of gradient descent when optimizing a model."
- "Alert you to the dangers of overfitting!"
keypoints:
- "A GPU needs to be available in order for you to use it."
- "Not all GPUs are the same."
---
 
 In this section we will establish the mathematical foundations of machine learning. We will define three important quantities: **data**, **models**, and **loss functions**. We will then discuss the optimization procedure known as **gradient descent**.
 
# Find out if a GPU is available
 
The first thing you need to know when you're thinking of using a GPU is whether there is actually one available. There are many ways of checking this in Python depending on which libraries you are intending to use with your GPU. The [GPUtil library](https://pypi.org/project/GPUtil/) available for pip installation provides simple methods to check. For example:

~~~
import GPUtil
GPUtil.getAvailable()
~~~
{: .language-python}

will return a list of available GPUs. However, many libraries also have built in functionality to check whether a GPU compatible with that library is available. For PyTorch this can be done using:

~~~
import torch
use_cuda = torch.cuda.is_available()
~~~
{: .language-python}

This command will return a boolean (True/False) letting you know if a GPU is available.


# Find out the specifications of the GPU(s)

There are a wide variety of GPUs available these days, so it's oftne useful to check the specifications of the GPU(s) that are available to you. For example, the following lines of code will tell you (i) which version of CUDA the GPU(s) support, (ii) how many GPUs there are available, (iii) for a specific GPU (here `0`) what kind of GPU it is, and (iv) how much memory it has available in total.

~~~
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
~~~
{: .language-python}


# Selecting a GPU to use

In PyTorch, you can use the `use_cuda` flag to specify which device you want to use. For example:

~~~
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ",device)
~~~
{: .language-python}

will set the device to the GPU if one is available and to the CPU if there isn't a GPU available. This means that you don't need to hard code changes into your code to use one or the other. If there are multiple GPUs available then you can specify a particular GPU using its index, e.g.

~~~
device = torch.device("cuda:2" if use_cuda else "cpu")
~~~
{: .language-python}


{% include links.md %}
