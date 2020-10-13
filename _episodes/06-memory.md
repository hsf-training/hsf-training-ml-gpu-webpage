---
title: "Memory considerations"
teaching: 10
exercises: 10
questions:
- "Why do I need to send my data to the GPU in batches?"
- "How can I monitor the GPU memory usage?"
objectives:
- "Prepare the dataset for machine learning."
- "Get excited for machine learning!"
keypoints:
- "One must properly format data before any machine learning takes place."
- "Data can be formatted using scikit-learn functionality; using it effectively may take time to master."
---

# Mini-batching

There are two reasons that we sub-divide the data into mini-batches during training:

* To produce a smoother loss curve;
* To make sure that the data fit into GPU memory.

> ## Wise Words
> *Friends don't let friends use mini-batches larger than 32* - Yann LeCunn
{: .callout}
