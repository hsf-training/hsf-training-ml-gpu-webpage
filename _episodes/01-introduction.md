---
title: "Introduction"
teaching: 15
exercises: 0
questions:
- "What is a GPU?"
- "Which machine learning libraries support GPUs?"
- "Why should I use a GPU for my ML code?"
objectives:
- "Discuss the general learning task in machine learning."
- "Provide examples of machine learning in high energy physics."
- "Give resources to people who want to become proficient in machine learning."
keypoints:
- "The 3 main tasks of Machine Learning are regression, classification and generation."
- "Machine learning has many applications in high energy physics."
- "If you want to become proficient in machine learning, you need to practice."
---

# What is a GPU?

Graphics Processing Units (GPUs) are specialised processors that contain many cores. They were originally designed to render graphics but these days they're used for other things as well. CPUs can also have multiple cores, but they don't have as many as a GPU. Typicaly GPUs will have 1000s of small cores on a single processor. The differences between CPUs and GPUs are summarised in this [table](https://blogs.nvidia.com/blog/2009/12/16/whats-the-difference-between-a-cpu-and-a-gpu/).

![GPU vs CPU](../plots/cpugpu_comp.png){:width="80%"}

It's important to note that just because GPUs have more cores than CPUs they are not universally better. There is a trade-off between increasing the number of cores and the flexibility of what they can be used for. GPU cores are smaller and more specialised than the cores on a CPU, this means that they are better for specific applications, but cannot be optimised or used efficiently in as many ways as CPUs. 

In particular GPUs are very efficient for performing highly parallel matrix multiplication, because this is an important application for graphics rendering. 

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



{% include links.md %}

