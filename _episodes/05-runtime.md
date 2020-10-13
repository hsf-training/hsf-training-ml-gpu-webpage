---
title: "Run time comparisons"
teaching: 15
exercises: 0
questions:
- "Will the ML performance of my model improve when I use the GPU?"
- "Will the computational performance of my model improve when I use the GPU?"
objectives:
- "Learn how to calculate run time when using a GPU."
- "Understand under which circumstances a GPU will improve performance."
keypoints:
- "Using a GPU will not improve your ML performance."
- "Using a GPU will improve your run time only under certain circumstances."
- "GPU processes are asynchronous."
---

# Model performance

Just as in the [Introduction to Machine Learning lesson]() you can evaluate the performance of your network using a variety of metrics. For example using

~~~
from sklearn.metrics import classification_report
# Random Forest Report
print (classification_report(y_test, y_pred,
                            target_names=["background", "signal"]))
~~~
{: .language-python}

However, remember that if you have made your predictions using the model on the GPU then you will need to move the prediction from your network `y_pred` off the GPU and onto the CPU before using the `classification_report()` function.

> ## Challenge
> Check the performance of the model you trained on the GPU and compare it to the same model trained on the CPU.
> 
> > ## Solution
> > You shouldn't see any difference in the performance of the two models. 
> {: .solution}
{: .challenge}

# Computational performance

Although there are different ways to evaluate the computational performance of your code, for the purpose of this tutorial the main metric that you're probably interested in is **run time**. 

### Calculating run time

An easy way to determine the run time for a particular section of code is to use the [Python time library](https://docs.python.org/3/library/time.html#time.time). 

~~~
import time
mytime = time.time()
print(mytime)
~~~
{: .language-python}

The `time.time()` function returns the time in seconds since January 1, 1970, 00:00:00 (UTC). By itself it's not always super useful, but for wrapping a piece of code and calculating the elapsed time between the start and end of that code it's a nice and simple method for determining run time.

~~~
import time
start = time.time()

# insert some code to do something here

end = time.time()
print("Run time [s]: ",end-start)
~~~
{: .language-python}


### Timing tests when using a GPU

When we are timing PyTorch processes that use a GPU it's necessary to add one extra line of code into this loop:

~~~
import time
start = time.time()

# insert some code to do something here

if use_cuda: torch.cuda.synchronize()    # <---------------- extra line
end = time.time()
print("Run time [s]: ",end-start)
~~~
{: .language-python}

This is because processes on a GPU run *asynchronously*. This means that when we send a process to the GPU it doesn't necessarily run immediately, instead it joins a queue. By calling the `torch.cuda.synchronize` function before specifying the `end` of our timing test, we can ensure that all of the processes on the GPU have actually run before we calculate the run time. 


> ## Challenge
> Calculate the run time for the training loop in your code.
> 
> > ## Solution
> > 
> > ~~~
> > model = model.to(device)
> >
> > start = time.time()
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
> >
> > if use_cuda: torch.cuda.synchronize()    
> > end = time.time()
> > print("Run time [s]: ",end-start)
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}





