---
title: "Is a GPU available"
teaching: 20
exercises: 0
questions:
- "How do I find out if a GPU is available?"
- "How can I determine the specifications of the GPU?"
- "How do I tell my code to run on the GPU?"
objectives:
- "Discuss the role of data, models, and loss functions in machine learning."
- "Discuss the role of gradient descent when optimizing a model."
- "Alert you to the dangers of overfitting!"
keypoints:
- "In a particular machine learning problem, one needs an adequate dataset, a reasonable model, and a corresponding loss function. The choice of model and loss function needs to depend on the dataset."
- "Gradient descent is a procedure used to optimize a loss function corresponding to a specific model and dataset."
- "Beware of overfitting!"
---
 
 In this section we will establish the mathematical foundations of machine learning. We will define three important quantities: **data**, **models**, and **loss functions**. We will then discuss the optimization procedure known as **gradient descent**.
 
# Find out if a GPU is available
 
~~~
import torch
use_cuda = torch.cuda.is_available()
~~~
{: .language-python}

# Find out the specifications of the GPU(s)

~~~
if use_cuda:
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__CUDA Device Name:',torch.cuda.get_device_name(0))
    print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
~~~
{: .language-python}


# Selecting a GPU to use

~~~
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print("Device: ",device)
~~~
{: .language-python}

{% include links.md %}
