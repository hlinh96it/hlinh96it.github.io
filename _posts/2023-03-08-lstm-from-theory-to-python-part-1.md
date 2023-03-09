---
title: Understanding Long Short Term Memory (LSTM) from Theory to Python - Part 1
author: hoanglinh
categories: [Deep Learning]
tags: [time series forecasting, natural language processing]
math: true
img_path: posts_media/2023-03-05-lstm-from-theory-to-python/
---

# Introduction

In the field of deep learning, recurrent neural networks (RNNs) have been used in sequential data analysis, where the current input **depends** on the previous inputs. However, traditional RNNs have difficulty retaining information from distant time steps, which limits their ability to learn long-term dependencies in the data. Technically, RNN (Recurrent Neural Network) suffers from vanishing gradient or exploding gradient problem, which refers to the difficulty of the network to propagate gradients effectively through time due to the nature of the **chain-like structure** of the network. 

To provide a more comprehensive explanation, let's delve more deeply into the concept of RNNs. You can explore my previous posts, which cover everything from the theory of RNNs to their implementation in Python, as well as the challenges they face when dealing with longer input sequences.

> **RNN - From theory to Python**: [https://www.notion.so/Recurrent-Neural-Network-From-Theory-to-Python-Code-9ab575122b7f4867a1d8d6fb96a7db45](https://www.notion.so/Recurrent-Neural-Network-From-Theory-to-Python-Code-9ab575122b7f4867a1d8d6fb96a7db45)
> 

# Theoretical Overview of LSTMs

**Long Short Term Memory (LSTM)** is a type of Recurrent Neural Network (RNN) that has been designed to overcome the vanishing gradient or exploding gradient problem in sequential data. This is particularly important in tasks where long-term dependencies are crucial. LSTM networks have shown great success in various applications such as speech recognition, machine translation, and natural language processing.

![rnn-lstm](rnn-lstm.png)_The darker the shade, the greater the sensitivity, thus standard recurrent network feel easier to “forget” the previous inputs_

LSTMs work by incorporating a memory cell that can selectively remember or forget information based on the input data. The memory cell is controlled by three gates: the input gate, the forget gate, and the output gate. The input gate decides which information to keep, the forget gate decides which information to discard, and the output gate decides which information to output. This allows LSTMs to selectively retain important information over time and discard irrelevant information.

![lstm-memory](lstm-diagram.png)_LSTM diagram. Source: [https://devopedia.org/long-short-term-memory](https://devopedia.org/long-short-term-memory)_

Moreover, LSTMs can also learn to **regulate the flow of information**, which is particularly useful in cases where the input data is noisy or irrelevant. This is achieved through the use of peephole connections, which allow the gates to take into account the previous state of the memory cell.

# The Math Behind LSTMs

## Forget gate: Allowed information to pass through the cell state

The forget gate $f_t$ is one of the three gates in an LSTM network that controls the flow of information through the memory cell. Its main purpose is to decide which information to discard from the memory cell. Specifically, the forget gate takes as input the previous hidden state $h_{t-1}$ and the current input $x_t$, and outputs a number between 0 and 1 for each element in the memory cell by using sigmoid activation function. A value of 0 means "**forget this information**" while a value of 1 means "**keep this information**" The forget gate is designed to selectively remove irrelevant information from the memory cell, allowing the LSTM to retain only the most important information.

![forget-gate.png](forget-gate.png)

## Update/Input gate: Information to be stored in the cell state

The input gate $i_t$ is another gate in the LSTM network that determines which information is stored in the memory cell. The input gate takes as input the previous hidden state $h_{t-1}$ and the current input $x_t$, and outputs a number between 0 and 1 for each element in the memory cell by using a sigmoid activation function. A value of 0 means "**discard this information**" while a value of 1 means "**store this information**". The input gate is responsible for selectively storing relevant information in the memory cell, while discarding irrelevant information.

![memory-gate.png](memory-gate.png)

<aside>
💡 The forget gate and input gate in an LSTM network perform different functions. The forget gate determines which information to discard from the memory cell, while the input gate determines which information to store in the memory cell.

</aside>

## Update previous cell state

This is the time to update an previous cell state $C_{t-1}$ to a new state $C_t$. The previous steps decided what to do, and at this step just do it.

![update-gate.png](update-gate.png)

## Output gate

The output gate in an LSTM network controls the flow of information from the memory cell to the output. The output gate takes as input the previous hidden state $h_{t-1}$, the current input $x_t$, and the current memory cell state $C_t$, and outputs a number between 0 and 1 for each element in the memory cell by using a sigmoid activation function. The output gate is responsible for selectively outputting relevant information from the memory cell, while discarding irrelevant information.

![output-gate.png](output-gate.png)

# Applications of LSTMs

LSTMs have a wide range of practical applications in various fields. In natural language processing, LSTMs can be used for language modeling, sentiment analysis, and machine translation. In speech recognition, LSTMs can be used for phoneme classification and keyword spotting. In finance, LSTMs can be used for stock price prediction and anomaly detection.

LSTMs can also be used for generating new content. For example, LSTMs can be trained on a dataset of Shakespearean sonnets and used to generate new sonnets that are similar in style and content. LSTMs can also be used for generating music or creating new images.

# Challenges of LSTMs

However, there are also limitations to LSTMs. One of the main challenges is selecting the right hyperparameters, such as the number of LSTM units, the learning rate, and the number of epochs. Another challenge is dealing with vanishing gradients, which can occur when the gradients become too small during backpropagation. This can result in slow convergence or even a complete halt in learning.

# Conclusion

LSTMs are a powerful type of RNN that can capture long-term dependencies in sequential data. Understanding the theory and math behind LSTMs is crucial to implementing them effectively in code. With the help of deep learning libraries such as Keras and TensorFlow, implementing LSTMs has become more accessible than ever before.

By understanding how LSTMs work and their various applications, you can leverage this powerful tool in your own projects. However, it's important to keep in mind the challenges and limitations of LSTMs and to experiment with different hyperparameters to achieve optimal performance.

# References

[1] [https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11](https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11)

[2] [https://devopedia.org/long-short-term-memory](https://devopedia.org/long-short-term-memory)

[3] [https://pub.towardsai.net/building-a-lstm-from-scratch-in-python-1dedd89de8fe](https://pub.towardsai.net/building-a-lstm-from-scratch-in-python-1dedd89de8fe)