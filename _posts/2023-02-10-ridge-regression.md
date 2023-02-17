---
title: Thuật toán Ridge Regression in Python
author: hoanglinh
categories: [Machine Learning]
tags: [regression model]
math: true
img_path: posts_media/2023-02-18-posts/
---

## Giới thiệu thuật toán Ridge Regression

Như chúng ta đã biết, khi training các thuật toán machine learning, chúng ta có thể dễ dàng gặp phải hiện tượng overfitting hoặc underfitting. Overfitting là hiện tượng model cho ra kết quả accuracy rất cao ở tập train nhưng lại rất thấp ở tập validation hoặc test. Điều này có thể lý giải bởi việc model quá phức tạp so với đặc tính của training data. Trong thực tế, điều quan trọng nhất đối với các machine learning model là nó có khả năng khái quát hóa các đặc tính của data và từ đó dự đoán được đúng xu hướng của data trong khi vẫn đạt được độ chính xác nhất định.

![polynomial_curves](polynomial_curves.png)_Mức độ phức tạp của mô hình theo sự thay đổi của bậc_

Hình trên diễn tả mức độ phức tạp của model khi degree tăng lên, degree càng cao thì độ phức tạp càng cao. Nếu ta sử dụng degree quá cao có thể dễ dàng đạt được accracy cao ở tập train nhưng thực tế là đã mất tính tổng quát của cả tập data. 

Nếu để ý, ta có thể nhận thấy rằng nếu ta cho giá trị weight khác nhau cho hệ số của phương trình, ta có thể thay đổi đường đi của đồ thị đó. Nói rõ hơn, nếu ta đặt $w_3 \rightarrow 0$ đối với degree = 3, đồ thị cho ra sẽ có dạng phương trình bậc 2. Điều này giúp giảm độ phức tạp của đồ thị và từ đó có thể giúp model mặc dù giảm độ chính xác nhưng lại có tính tổng quát cao hơn, từ đó accuracy ở tập validation và test cao hơn.

> Như vậy kiểm soát độ lớn của hệ số ước lượng, đặc biệt là với bậc cao, sẽ giúp giảm bớt mức độ phức tạp của mô hình và thông qua đó khắc phục hiện tượng *quá khớp* (overfitting). Vậy làm cách nào để kiểm soát chúng, cùng tìm hiểu một trong những thuật toán Regression có khả năng làm được điều đó, đó là Ridge Regression.
{: .prompt-info}

## Sự thay đổi của hàm mất mát trong hồi qui Ridge

Về cơ bản, thuật toán Ridge Regression là một biến thể khác của Linear Regression, trong đó ta thêm một thành phần hiệu chỉnh (regularization) vào hàm tính toán loss. Điều này không những giúp model có thể mô tả được pattern của data tốt hơn mà còn giữ các tham số của model nhỏ nhất có thể. 
$$
\begin{split}\begin{eqnarray} \mathcal{L}(\mathbf{w}) & = & \frac{1}{N}||\bar{\mathbf{X}}\mathbf{w} - \mathbf{y}||_{2}^{2} + \alpha ||\mathbf{w}||_2^2 \\
& = & \frac{1}{N}||\bar{\mathbf{X}}\mathbf{w} - \mathbf{y}||_{2}^{2} + \underbrace{\alpha R(\mathbf{w})}_{\text{regularization term}}
\end{eqnarray}\end{split}
$$

> Thành phần regularization chỉ được thêm vào hàm loss khi training, khác với hàm đánh giá kết quả prediction. Ví dụ như sử dụng hàm **log loss** nhưng lại sử dụng **precision/recall** đối với bài toán classification. Ta cần lưu ý rằng sử dụng đơn vị đánh giá cần phù hợp với objective của từng bài toán để đạt được mục đích.
{: .prompt-tip}
