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

**If you are: “Just keep me the code 😤”, please check it: [Understanding Long Short Term Memory (LSTM) from Theory to Python - Part 2](https://www.notion.so/Understanding-Long-Short-Term-Memory-LSTM-from-Theory-to-Python-Part-2-ec36d1db3ad34ca6a6083df305b6b697).** In Part 2, we'll start by reviewing the theoretical concepts we covered in Part 1, and then delve into the practical applications of LSTMs. We'll explore the basics of coding LSTM from scratch and then move on to using Tensorflow to implement LSTMs in your projects. This post will be a comprehensive guide that covers everything you need to know to get started with LSTMs.

You'll learn how to build and train your own LSTM network, and we'll cover a range of topics, including data preprocessing, model selection, and hyperparameter tuning. We'll also explore how to use LSTM networks for a variety of practical applications, such as time series prediction, text classification, and image recognition.

Whether you're just starting out with LSTMs, or you're an experienced deep learning practitioner looking to expand your skills, my next post will provide you with the knowledge and tools you need to take your projects to the next level. So be sure to check it out!

In addition, you can also delve more deeply into the concept of RNNs in my previous posts, which cover everything from the theory of RNNs to their implementation in Python, as well as the challenges they face when dealing with longer input sequences.

> RNN - From theory to Python: [https://www.notion.so/Recurrent-Neural-Network-From-Theory-to-Python-Code-9ab575122b7f4867a1d8d6fb96a7db45](https://www.notion.so/Recurrent-Neural-Network-From-Theory-to-Python-Code-9ab575122b7f4867a1d8d6fb96a7db45)
{: .prompt-info}

# Theoretical Overview of LSTMs

**Long Short Term Memory (LSTM)** is a type of Recurrent Neural Network (RNN) that has been designed to overcome the vanishing gradient or exploding gradient problem in sequential data. This is particularly important in tasks where long-term dependencies are crucial. LSTM networks have shown great success in various applications such as speech recognition, machine translation, and natural language processing.

![rnn-lstm](rnn-lstm.png)_The darker the shade, the greater the sensitivity, thus standard recurrent network feel easier to “forget” the previous inputs_

LSTMs operate by using a memory cell that can remember or forget information based on the input data. The memory cell is regulated by three gates: the input gate, the forget gate, and the output gate. The input gate determines which information to retain, the forget gate determines which information to discard, and the output gate determines which information to output. In this way, LSTMs make decisions by considering the current input, previous output, and previous memory. They produce a new output and modify their memory accordingly. This mechanism allows LSTMs to selectively retain important information and discard irrelevant information over time.

![lstm-memory](lstm-diagram.png)_LSTM diagram. Source: [https://devopedia.org/long-short-term-memory](https://devopedia.org/long-short-term-memory)_

| Term | Definition |
| --- | --- |
| Memory Cell | The component of an LSTM network that stores information over time and regulates <br> the flow of information with three gates: the input gate, the forget gate, and the output gate. |
| Input Gate | Stores relevant information in the memory cell. |
| Forget Gate | Removes irrelevant information from the memory cell. |
| Output Gate | Outputs relevant information from the memory cell. |
| Units | The individual memory cells or neurons that process input data and produce output in <br> an LSTM layer.  |


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

This is the time to update an previous cell state $C_{t-1}$ to a new state $C_t$. The input is the old memory (a vector), $C_{t-1}$. The first cross ✖ it passes through is the forget valve. It is actually an **element-wise multiplication** operation. So if you multiply the old memory $C_{t-1}$ with a vector that is close to 0, that means you want to forget most of the old memory. You let the old memory goes through, if your forget valve equals 1.

Then the second operation the memory flow will go through is this + operator. This operator means **piece-wise summation**. New memory and the old memory will merge by this operation. How much new memory should be added to the old memory is controlled by another valve, the ✖ below the + sign.

![update-gate.png](update-gate.png)

## Output gate

The output gate in an LSTM network controls the flow of information from the memory cell to the output. The output gate takes as input the previous hidden state $h_{t-1}$, the current input $x_t$, and the current memory cell state $C_t$, and outputs a number between 0 and 1 for each element in the memory cell by using a sigmoid activation function. The output gate is responsible for selectively outputting relevant information from the memory cell, while discarding irrelevant information.

![output-gate.png](output-gate.png)

## What is the number of units in an LSTM cell?

The number of units in an LSTM cell refers to the number of memory cells or neurons in the LSTM layer. Each unit in the LSTM layer processes input data and produces output, and the overall performance of the LSTM network depends on the number of units and how they are connected.

![lstm-unit.png](lstm-unit.png){: width="700"}_Source: [https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/](https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/)_


# Applications of LSTMs

LSTMs have a wide range of practical applications in various fields. In natural language processing, LSTMs can be used for language modeling, sentiment analysis, and machine translation. In speech recognition, LSTMs can be used for phoneme classification and keyword spotting. In finance, LSTMs can be used for stock price prediction and anomaly detection.

For instance, LSTMs can be trained on a dataset of Shakespearean sonnets, which can be used to generate new sonnets having similar style and content. Similarly, LSTMs can be used for creating new pieces of music or generating new images. By training LSTMs on a dataset of existing content, it is possible to develop a model that can produce new content in the same style and format, providing an effective and efficient way to generate new ideas and creative works.

# Challenges of LSTMs

However, while LSTMs have proven to be effective in various applications, they are not without limitations. One of the primary challenges associated with LSTMs is selecting the optimal hyperparameters. This requires thorough experimentation and testing, as the number of LSTM units, learning rate, and number of epochs can all have a significant impact on the model's performance. Additionally, vanishing gradients can be a major obstacle when training an LSTM. This can occur when the gradients become too small during backpropagation, resulting in slow convergence or even a complete halt in learning. To address this issue, researchers have developed a number of techniques, such as gradient clipping and layer normalization, to ensure that the gradients remain within a reasonable range. Despite these challenges, LSTMs remain one of the most widely used and effective types of recurrent neural networks for sequential data processing tasks.

# Conclusion

LSTMs are a powerful type of RNN that can capture long-term dependencies in sequential data. Understanding the theory and math behind LSTMs is crucial to implementing them effectively in code. With the help of deep learning libraries such as Keras and TensorFlow, implementing LSTMs has become more accessible than ever before.

By understanding how LSTMs work and their various applications, you can leverage this powerful tool in your own projects. However, it's important to keep in mind the challenges and limitations of LSTMs and to experiment with different hyperparameters to achieve optimal performance.

> **If you are: “Just keep me the code 😤”, please check it: [Understanding Long Short Term Memory (LSTM) from Theory to Python - Part 2](https://www.notion.so/Understanding-Long-Short-Term-Memory-LSTM-from-Theory-to-Python-Part-2-ec36d1db3ad34ca6a6083df305b6b697).**
{: .prompt-warning}

# References

[1] [https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11](https://towardsdatascience.com/the-vanishing-exploding-gradient-problem-in-deep-neural-networks-191358470c11)

[2] [https://devopedia.org/long-short-term-memory](https://devopedia.org/long-short-term-memory)

[3] [https://pub.towardsai.net/building-a-lstm-from-scratch-in-python-1dedd89de8fe](https://pub.towardsai.net/building-a-lstm-from-scratch-in-python-1dedd89de8fe)

[4] [https://mmuratarat.github.io/2019-01-19/dimensions-of-lstm](https://mmuratarat.github.io/2019-01-19/dimensions-of-lstm)