---
title: Batch Normalization trong Convolutional Neural Networks - Tối ưu hiệu suất và ổn định
author: hoanglinh
categories: [Deep Learning]
tags: [classification, convolution neural networks]
math: true
---

Batch Normalization [1] (chuẩn hóa theo batch) là một kỹ thuật quan trọng trong Convolutional Neural Networks (CNNs) giúp cải thiện hiệu suất và ổn định trong việc huấn luyện mô hình. Trong bài viết này, chúng ta sẽ khám phá lý do tại sao chúng ta cần sử dụng Batch Normalization và tìm hiểu về lợi ích và cách hoạt động của nó. Chúng ta sẽ đi sâu vào phương pháp Batch Normalization cùng với ví dụ code sử dụng thư viện numpy và áp dụng nó vào bài toán phân loại chó mèo.

# Lợi ích của Batch Normalization

Batch Normalization có nhiều lợi ích quan trọng khi áp dụng vào CNNs. Dưới đây là một số lợi ích chính:

1. Ổn định quá trình huấn luyện: Batch Normalization giúp ổn định quá trình huấn luyện bằng cách chuẩn hóa đầu ra của các lớp trước đó trong quá trình lan truyền thuận. Điều này giúp giảm hiện tượng gradient mất mát và gradient tăng vượt quá mức cho phép, đồng thời tăng tốc độ hội tụ của quá trình tối ưu hóa.
2. Giảm overfitting: Batch Normalization có khả năng giảm overfitting trong quá trình huấn luyện. Kỹ thuật này áp dụng một phương pháp nhỏ nhẹ của regularization bằng cách điều chỉnh trung bình và độ lệch chuẩn trong quá trình huấn luyện, giúp mô hình tổng quát hóa tốt hơn trên dữ liệu kiểm tra.
3. Tăng tốc độ huấn luyện: Batch Normalization giúp tăng tốc độ huấn luyện bằng cách giảm sự phụ thuộc của mô hình vào các giá trị khởi tạo ban đầu và tăng khả năng hội tụ của quá trình tối ưu hóa. Điều này cho phép chúng ta sử dụng tỷ lệ học tập lớn hơn và giảm thời gian huấn luyện tổng thể.

# Cách hoạt động của Batch Normalization

Batch Normalization hoạt động dựa trên hai bước chính: chuẩn hóa batch và điều chỉnh.

1. Chuẩn hóa batch: Đầu tiên, trung bình và độ lệch chuẩn của các giá trị đầu ra trong mỗi batch được tính toán. Sau đó, các giá trị đầu ra của batch được chuẩn hóa bằng cách trừ đi trung bình và chia cho độ lệch chuẩn.
2. Điều chỉnh: Sau khi chuẩn hóa, các giá trị đầu ra được thay đổi bằng cách nhân với một tham số "scale" và cộng với một tham số "shift". Điều này cho phép mô hình học cách điều chỉnh và tối ưu hóa phân phối của các giá trị đầu ra.

Phương trình chuẩn hóa batch: 

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Phương trình điều chỉnh:

$$
y = \gamma \hat{x} + \beta
$$

Trong đó:

- $x, \hat{x}$ là giá trị đầu vào và sau khi chuẩn hóa
- $\mu, \sigma$ là giá trị trung bình và độ lệch chuẩn của `batch`
- $\epsilon$ là hằng số nhỏ để tránh trường hợp phép chia cho 0
- $\beta,\gamma$ là các tham số có thể học và update để model có độ chính xác cao hơn

Code ví dụ sử dụng Numpy:

```python
import numpy as np

def batch_normalization(x, epsilon=1e-8):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    normalized_x = (x - mean) / np.sqrt(std ** 2 + epsilon)
    return normalized_x

# Sử dụng batch normalization trên tập dữ liệu huấn luyện
normalized_train_data = batch_normalization(train_data)

# Sử dụng batch normalization trên tập dữ liệu kiểm tra
normalized_test_data = batch_normalization(test_data)
```

Trên đây là một phần nhỏ về Batch Normalization và cách sử dụng nó trong CNNs. Batch Normalization là một kỹ thuật quan trọng để tăng hiệu suất và ổn định trong huấn luyện mô hình. Bằng cách áp dụng Batch Normalization, chúng ta có thể giảm overfitting, tăng tốc độ huấn luyện và đạt được kết quả tốt hơn trên tập dữ liệu kiểm tra.

# References
1. Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." International conference on machine learning. pmlr, 2015.
2. [https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338](https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338)