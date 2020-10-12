---
title: "Performance comparisons"
teaching: 15
exercises: 0
questions:
- "Will the ML performance of my model improve when I use the GPU?"
- "Will the computational performance of my model improve when I use the GPU?"
objectives:
- "Provide links to tutorials and textbooks that will help you get better at Python."
- "Provide links to machine learning library documentation."
keypoints:
- "NumPy and pandas are the main libraries for scientific computing."
- "scikit-learn and TensorFlow are two good options for machine learning in Python."
---

# Model performance


# Computational performance

Although there are different ways to evaluate the computational performance of your code, for the purpose of this tutorial the main metric that you're probably interested in is **run time**. 

~~~
import time
mytime = time.time()
print(mytime)
~~~
{: .language-python}

~~~
import time
start = time.time()

# insert some code to do somthing here

end = time.time()
print("Run time [s]: ",end-start)
~~~
{: .language-python}


### Timing tests when using a GPU

~~~
import time
start = time.time()

# insert some code to do somthing here

if use_cuda: torch.cuda.synchronize()
end = time.time()
print("Run time [s]: ",end-start)
~~~
{: .language-python}


# Code Example





