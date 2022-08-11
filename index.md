---
layout: lesson
root: .  # Is the only page that doesn't follow the pattern /:path/index.html
permalink: index.html  # Is the only page that doesn't follow the pattern /:path/index.html
---
This tutorial explores Machine Learning using scikit-learn and TensorFlow for applications in high energy physics.

Extended from a [version developed by Luke Polson for the 2020 USATLAS Computing Bootcamp](https://lukepolson.github.io/HEP_ML_Lessons/).

<!-- this is an html comment -->

{% comment %} This is a comment in Liquid {% endcomment %}


> ## Prerequisites
> * A [Kaggle](https://www.kaggle.com/) account. Click [here to create an account](https://www.kaggle.com/account/login?phase=startRegisterTab&returnUrl=%2F)
> * Basic Python knowledge, e.g. through the [Software Carpentry Programming with Python lesson](https://swcarpentry.github.io/python-novice-inflammation/) 
{: .prereq}

Introduction
------------

For physicists working on analysis in data-intensive fields such as particle physics, it's quite common these days to start developing new machine learning applications. But many machine learning applications run more efficiently on GPU.

The aim of this lesson is to:
- demonstrate how to move an existing machine learning model onto a GPU
- discuss some of the common issues that come up when using machine learning applications on GPUs

> ## The skills we'll focus on:
>
> 1.  Understanding a bit about GPUs
> 2.  Using Python & PyTorch to discover what kind of GPU is available to you 
> 3.  Moving a machine learning model onto the GPU
> 4.  Comparing the performance of the machine learning model between the CPU and the GPU
{: .checklist}

{% include curriculum.html %}

Videos are provided at the top of each page to help guide you. For the Introduction section, which has no coding, the video simply takes you through the text, so choose whichever way you learn best: video or reading. For the remaining sections, the videos take you through the coding live.

{% include links.md %}
