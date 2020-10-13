---
title: "Introduction"
teaching: 15
exercises: 0
questions:
- "What is a GPU?"
- "Which machine learning libraries support GPUs?"
- "Why should I use a GPU for my ML code?"
objectives:
- "Discuss the differences between CPUs and GPUs."
- "Provide examples of Python machine learning libraries that support GPUs."
keypoints:
- "GPUs are great for highly-parallel processing."
- "CPUs are more flexible than GPUs."
- "GPUs are most useful for neural network applications in machine learning."
---

# What is a GPU?

A processor *core* is an individual processor within a Central Processing Unit (CPU). This core is the computer chip inside the CPU that performs calculations. Today nearly all computers have *multi-core processors*, which means that their CPU contains more than one core. This increases the performance of the CPU because it can do more calculations. Confusingly, the terms *processor* and *core* often get used interchangeably.

Graphics Processing Units (GPUs) are specialised processors that contain many cores. They were originally designed to render graphics but these days they're used for other things as well. Although CPUs can have multiple cores they don't have nearly as many as a GPU. Typicaly GPUs will have 1000s of small cores on a single processor. The differences between CPUs and GPUs are summarised in this [table](https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/).

![GPU vs CPU](../plots/cpugpu_comp.png){:width="80%"}

It's important to note that just because GPUs have more cores than CPUs they are not universally better. There is a trade-off between increasing the number of cores and the flexibility of what they can be used for. GPU cores are smaller and more specialised than the cores on a CPU, this means that they are better for specific applications, but cannot be optimised or used efficiently in as many ways as CPUs. 

In particular GPUs are very efficient for performing highly parallel matrix multiplication, because this is an important application for graphics rendering. 

### CUDA vs. OpenCL

The Compute Unified Device Architecture (CUDA) is a parallel computing platform developed by NVIDIA that enables software to jointly use both the CPU and GPU. OpenCL is an alternative and more general parallel computing platform developed by Apple that allows software to access not only CPU and GPU simultaneously, but also FPGAs and other DSPs. Both CUDA and OpenCL are compatible with Python and allow the user to implement highly parallel processes on a GPU without needing to explicitly specify the parallelisation, i.e. they do the optimisation for you. 

### AMD vs. NVIDIA

While both AMD and NVIDIA are major vendors of GPUs, NVIDIA is currently the most common GPU vendor for machine learning and cloud computing. Most GPU-enabled Python libraries will only work with NVIDIA GPUs.

# Which Python machine learning libraries support GPUs?

* Tensorflow
* PyTorch
* Keras
* Caffe

An important ML Python library that you may notice is missing from this list is [Scikit-Learn](https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support). Scikit-learn does not support GPU processing at the moment and there are currently no plans to implement support in the near future. Why is this? Well, GPU suport is primarily used for neural networks and deep learning, neither of which are key elements of the Scikit-learn library.

# Why should I use a GPU for my ML code?

The matrix operations that GPus are optimised for are exactly what happens in the training step for building a deep learning model. In a neural network, the process of multiplying input data by weights can be formulated as a matrix operation and as your network grows to include 10s of millions of parameters it also becomes a pretty big one. Having many cores available to perform this matrix multiplication in parallel means that the GPU can quickly outperform a CPU in this case. 

However, if you're *not* using a neural network as your machine learning model you may find that a GPU doesn't improve the computation time. It's the large matrix multiplications required for neural networks that really make GPUs useful. Likewise if you are using a neural network but its very small then again a GPU will not be any faster than a CPU - in fact it might even be slower. 


{% include links.md %}

