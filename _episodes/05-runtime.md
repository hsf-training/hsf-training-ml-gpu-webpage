---
title: "Run time comparisons"
teaching: 15
exercises: 0
questions:
- "Will the ML performance of my model improve when I use the GPU?"
- "Will the computational performance of my model improve when I use the GPU?"
objectives:
- "Provide links to tutorials and textbooks that will help you get better at Python."
- "Provide links to machine learning library documentation."
keypoints:
- "Using a GPU will not improve your ML performance."
- "Using a GPU will improve your run time only under certain circumstances."
- "GPU processes are asynchronous."
---

# Model performance


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





