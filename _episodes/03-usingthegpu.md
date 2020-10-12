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

# Neural Network Theory Introduction

model = Classifier_MLP(in_dim=input_size, hidden_dim=hidden_size, out_dim=num_classes).to(device=device)
