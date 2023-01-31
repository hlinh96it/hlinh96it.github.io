---
title: Thuật toán Gradient Descent (GD) với Python
author: hoanglinh
categories: [Machine Learning, Deep Learning]
tags: [optimization algorithms]
math: true
img_path: posts_media/2023-01-07-posts/
---

## Giới thiệu về thuật toán Gradient Descent (GD)

Thông thường, các thuật toán machine learning (ML) sẽ sử dụng các attributes hay còn gọi là features trong dataset để học và dự đoán output $y$. Tuy nhiên, có thể một vài features thể hiện nhiều thông tin hơn so với các features khác. Ví dụ như giá nhà chủ yếu dựa vào features về khoảng cách so với trung tâm, nhà trong ngõ hay mặt đường, vv. Rõ ràng, ta phải sắp xếp mức độ quan trọng của từng feature sao cho ML model hiểu và tập trung vào học các features đó. Từ đó sinh ra khái niệm weights of features, weight lớn đồng nghĩa với feature đó quan trọng. 

> Câu hỏi đặt ra là làm sao tìm được optimal weights sao cho ML model có thể dự đoán được chính xác output nhất. Gradient descent được sinh ra để giải quyết vấn đề này.
{: .prompt-info}

Gradient descent (GD) là thuật toán tối ưu nhằm tìm kiếm optimal weights của các features cho ML model. Hiểu cơ bản, đầu tiên ta chọn ngẫu nhiên weight cho các features, predict output thông qua hàm `forward()`, tính loss bằng cách so sánh giá trị predict với actual, và cuối cùng là update weight bằng cách tính đạo hàm riêng của từng weight với loss. Loss càng nhỏ càng chứng tỏ là ML model predict càng gần với actual. Có nhiều phương pháp tính loss, có thể kể đến là Mean square error (MSE):

$$
{\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum_{i=1}^{n}\left(Y_{i}-{\hat {Y_{i}}}\right)^{2}}\tag{1}
$$

Trong đó, $n$ là số lượng sample dùng để tính loss, $\hat{Y}_i$ là giá trị predict cho sample $i$. $\hat{Y}_i$ có thể được tính bởi công thức:
$$
\hat{Y}_i = \beta + \theta X_i \tag{2}
$$
Trong đó, $\beta$ là bias cho model, $\theta$ là weight vector cho từng feature của input $X_i$.

Gradient descent là thuật toán tối ưu thông qua vòng lặp (iterative optimization) với mục tiêu là update các tham số $\theta$ sao cho giá trị loss giảm dần theo số vòng lặp. GD sẽ dừng khi gặp điều kiện kết thúc như loss giảm không đáng kể nữa hoặc kết thúc số lần lặp. Giá trị của các tham số $\theta$ được update theo đạo hàm riêng với hàm loss và learning rate:

$$
\theta^{t+1} =\theta - \eta  ∇_\theta
$$

Trong đó, $∇_\theta$ là ký hiệu chỉ vector của các đạo hàm riêng của $\theta$ theo hàm loss.

### Learning rate $\eta$

Trong công thức update $\theta$, có một tham số quan trọng nữa là $\eta$ -learning rate, đó là giá trị chỉ độ lớn hay mức độ update các parameters sau mỗi vòng lặp. Nếu $\eta$ nhỏ thì mất nhiều thời gian để tìm được optimal parameter $\theta$, $\eta$ lớn thì thuật toán khó hội tụ. Để tìm được learning rate $\eta$ phù hợp, ta cần thực nghiệm các tham số phù hợp, ví dụ như hình dưới.

![gradient-descent](gradient-descent.png)_Sự ảnh hưởng của learning rate đến mô hình_

Bên cạnh đó, hàm loss của các ML model có thể là hàm đa thức, túc là đồ thị loss không chỉ đơn thuần là hàm bậc 2 như trên, MSE loss. Ví dụ như hình dưới, đây là hàm loss ở dạng đa thức. Với loss dạng này càng làm tăng mức độ quan trọng của việc chọn learning rate, nếu quá nhỏ, model có thể bị “kẹt” ở **local minimum**, hoặc mất rất nhiều thời gian để đi đến **global minimum**.

![local-vs-global-minimum](local-vs-absolute-extrema.png){: width="500"}_Local và Global minimum_

### Feature scaling

Trên thực tế, hàm chi phí có dạng đồ thị giống chiếc bát, nếu các feature (input - thành phần của vector X) có cùng phạm vi giá trị, thì miệng bát sẽ tròn và để GD đi xuống đáy bát sẽ nhanh hơn. Nếu các feature khác phạm vi giá trị thì miệng bát sẽ bị kéo dài ra và việc đi xuống đáy bát sẽ tốn thời gian hơn. Đây là lý do vì sao các feature của vector đầu vào X cần phải được scaling (căn chỉnh).

![feature-scaling](gd-with-wo-feature-scaling.png){: width="700"}_Gradient Descent with (left) and without (right) feature scaling_

Như bạn có thể thấy, ở bên phải thuật toán Gradient Descent đi thẳng về điểm tối thiểu, do đó nhanh chóng đạt được cực tiểu toàn cục, trong khi bên trái, nó đi theo hướng gần như trực giao với hướng về cực thiểu toàn cục, vì vậy nó kết thúc bằng 1 hành trình dài xuống một 1 mặt gần như bằng phẳng. Cuối cùng nó sẽ đạt đến mức cực tiểu, nhưng sẽ mất nhiều thời gian.

>  Khi bạn thực hiện thuật toán Gradient Descent, bạn nên đưa các feature về cùng phạm vi giá trị (sử dụng `StandardScaler` của thư viện **Scikit-Learn**)
{: .prompt-info}

## Batch Gradient Descent



## Referenes

1. [Gradient Descent Tiếng Việt](https://ndquy.github.io/posts/gradient-descent-2/)