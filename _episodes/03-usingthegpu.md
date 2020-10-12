---
title: "Using the GPU"
teaching: 5
exercises: 15
questions:
- "How do I send my data to the GPU?"
- "How do I train my model on the GPU?"
objectives:
- "Learn how to move data between the CPU and the GPU."
- "Be able to identify common errors when moving data."
keypoints:
- "Both the model and the data must be moved onto the GPU for training."
- "Data should be moved onto the GPU in batches." 
---

Once you have selected which device you want PyTorch to use then you can specify which parts of the computation are done on that device. Everything will run on the CPU as standard, so this is really about deciding which parts of the code you want to send to the GPU. For a neural network, training a model is typically the most computationally expensive part of your code and so that's where GPUs are normally utilised. To run a training loop in this way requires that two things are passed to the GPU: (i) the model itself and (ii) the training data.


# Sending the model to the GPU

In order to train a model on the GPU it is first necessary to send the model itself to the GPU. This is necessary because the trainable parameters of the model need to be on the GPU so that they can be applied and updated in each forward-backward pass. In PyTorch sending the model to the GPU is very simple:

~~~
model = model.to(device=device)
~~~
{: .language-python}

You can also do this when you initialise your model. For the example from the ML tutorial this would look like:

~~~
model = Classifier_MLP(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes).to(device=device)
~~~
{: .language-python}

> ## Older PyTorch versions
> In older PyTorch versions, sending things to the GPU was specified in a less flexible way. Instead of using the `.to(device=device)` syntax, one used `.cuda()` to send things to the GPU and `.cpu()` to send things to the CPU. Although this is deprecated it will still work with more recent versions of PyTorch, and is often seen in older tutorials.
{: .callout}

# Sending the Data to the GPU

The second requirement for running the training loop on the GPU is to move the training data. This can be done in exactly the same way as for the model, i.e.

~~~
x_train, y_train = x_train.to(device), y_train.to(device)
~~~
{: .language-python}

Due to the memory limitations of GPUs compared with CPUs, the data should be moved in *mini-batches*, i.e. you shouldn't send your whole training data set to the GPU at the beginning of your code. Instead you should only send the data within a single batch iteratively during the training. 

> ## Challenge
> Adapt the training loop from the ML tutorial to use the GPU.
> 
> > ## Solution
> > 
> > ~~~
> > model = model.to(device)
> > for batch, (x_train, y_train) in enumerate(train_loader):
> >         
> >         x_train, y_train = x_train.to(device), y_train.to(device)
> >         
> >         model.zero_grad()
> >         pred, prob = model(x_train)
> >         
> >         acc = (prob.argmax(dim=-1) == y_train).to(torch.float32).mean()
> >         train_accs.append(acc.mean().item())
> >         
> >         loss = F.cross_entropy(pred, y_train)
> >         train_loss.append(loss.item())
> >        
> >         loss.backward()
> >         optimizer.step()
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Remember that if your model is on the GPU then the data in your validation loop will also need to be sent to the GPU. Otherwise you will see an error that looks like this:

~~~

~~~
{: .language-bash}


### Using the DataLoader Class with the GPU

If you are using the PyTorch `DataLoader()` class to load your data in each training loop then there are some keyword arguments you can set to speed up the data loading on the GPU. These should be passed to the class when you set up the data loader.

~~~
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
~~~
{: .language-python}

*Pinned memory* is used as a staging area for data transfers between the CPU and the GPU. By setting `pin_memory=True` when we initialise the data loader we are directly allocating space in pinned memory. This avoids the time cost of transfering data from the CPU to the pinned staging area every time we move the data onto the GPU later in the code. You can read more about pinned memory on the [nvidia blog](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/). 

### GPU/CPU data mis-matches

Remember that once you have sent a particular set of data to the GPU, if you want to perform a calculation on those data using the CPU then you will need to move it back again. One of the most common errors you will see when using a GPU is a mismatch between the locations of different data being used in a function. This is what we saw above when the validation data were not moved onto the GPU.

You can find out which device your data are on at different points in the code by using the `device` property:

~~~
print(x_train.device)
~~~
{: .language-python}

> ## Challenge
> Check which device the probability output from your model is held on. Do the same with the calculated loss.
> 
> > ## Solution
> > 
> > ~~~
> > for batch, (x_train, y_train) in enumerate(train_loader):
> >         
> >         x_train, y_train = x_train.to(device), y_train.to(device)
> >         
> >         model.zero_grad()
> >         pred, prob = model(x_train)
> >         print(prob.device)
> >         
> >         acc = (prob.argmax(dim=-1) == y_train).to(torch.float32).mean()
> >         train_accs.append(acc.mean().item())
> >         
> >         loss = F.cross_entropy(pred, y_train)
> >         train_loss.append(loss.item())
> >         print(loss.device)
> >
> >         loss.backward()
> >         optimizer.step()
> > ~~~
> > {: .language-python}
> > You should see that both the outputs from the model and the calculated loss are still on the GPU. If you want to use these values on the CPU you will need to use (e.g.)
> > ~~~
> > prob = prob.to('cpu')
> > loss = loss.to('cpu')
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}
