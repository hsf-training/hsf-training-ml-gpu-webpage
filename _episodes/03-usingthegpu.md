---
title: "Using the GPU"
teaching: 5
exercises: 15
questions:
- "How do I send my data to the GPU?"
- "How do I train my model on the GPU?"
objectives:
- "Examine the structure of a fully connected sequential neural network."
- "Look at the TensorFlow neural network Playground to visualize how a neural network works."
keypoints:
- "Neural networks consist of an input layer, hidden layers and an output layer."
- "TensorFlow Playground is a cool place to visualize neural networks!" 
---

Once you have specified which device you want PyTorch to use then you can specify which parts of the computation are done on that device. Everything will run on the CPU as standard, so this is really about specifying which parts of the code you want to send to the GPU. For a neural network, training a model is typically the most computationally expensive part of your code and so that's where GPUs are normally utilised. To run a training loop in this way requires that two things are passed to the GPU: (i) the model itself and (ii) the training data.


# Sending the model to the GPU

In order to train a model on the GPU it is first necessary to send the model itself to the GPU. This is necessary because the trainable parameters of the model need to be on the GPU so that they can be applied and updated in each forward-backward pass. In PyTorch sending the model to the GPU is very simple:

~~~
model = MyModel().to(device=device)
~~~
{: .language-python}

For the example from the ML tutorial this would look like:

~~~
model = Classifier_MLP(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes).to(device=device)
~~~
{: .language-python}

> ## Older PyTorch versions
> In older PyTorch versions, sending things to the GPU was specified in a less flexible way. Instead of using the `.to(device=device)` syntax, one used `.cuda()` to send things to the GPU and `.cpu()` to send things to the CPU. Although this is deprecated it will still work with more recent versions of PyTorch, and is often seen in older tutorials.
{: .callout}

# Sending the Data to the GPU

~~~
x_train, y_train = x_train.to(device), y_train.to(device)
~~~
{: .language-python}

> ## Challenge
> Adapt the training loop from the ML tutorial to use the GPU.
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
