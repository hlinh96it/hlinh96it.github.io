---
title: Gradient Descent and Loss function in a Intuitive way
author: hoanglinh
categories: [Machine Learning, Deep Learning]
tags: [optimization algorithms]
math: true
img_path: posts_media/2023-01-07-posts/
---

Trong bài viết này, chúng ta sẽ cùng nhau tìm hiểu về nguồn gốc cơ bản của thuật toán Gradient descent (GD) trong machine learning (ML) và hàm loss (tính toán mất mát) một cách cơ bản nhất. Giả sử rằng, bạn đã có kiến thức nền tảng cơ bản về hình học 2D hoặc 3D nhưng nếu không thì cũng không có vấn đề gì lớn.

Giả sử rằng, trong một mặt phẳng, ta được cho 3 điểm với tọa độ là $(1, 1), (2, 2), (3, 3)$, biểu diễn dưới mặt phẳng 2D, ta có hình như dưới:

![3-points](plot-3-points.png){: width="500"}

Bây giờ, nếu ta được yêu cầu là tìm phương trình thỏa mãn 3 điểm đó, hay nói cách khác là từ phương trình, ta có thể kẻ được một đường thẳng đi qua 3 điểm đó. Phương trình cần có dạng:

$$
y = f(x) = ax+ b
$$

Trong đó, $a$ và $b$ là hai hệ số của đường thẳng. Nhiệm vụ của chúng ta là tìm hệ số $a, b$ sao cho thỏa mãn phương trình. Trong trường hợp này, ta có thể nhẩm ngay ra nghiệm hay hệ số duy nhất của $a=0$ và $b=1$. Tuy nhiên, hãy thử với trường hợp khác xem sao:

![no-solution](3-lines-no-solution.png){: width="500"}

Trong trường hợp này, đường đỏ và xanh lam không thỏa mãn phương trình. Tuy nhiên, nó thể hiện được gần đúng giá trị của tất cả các điểm.  Điều này dẫn đến câu hỏi, nếu như không có nghiệm đúng, vậy ta có thể tìm được tham số $a,b$ nào mà đường “fit” thể hiện gần đúng nhất của tất cả điểm không? 

>  Cost function hay loss function là hàm được sử dụng để đánh giá mức độ sai lệch của đường “fit” với tất cả các điểm quan tâm.
{: .prompt-info}

## Tính toán Cost function

Về cơ bản, cost function hay hàm chi phí có nghĩa là đường dự đoán của ML model cách các điểm thực tế  bao xa. Nói cách khác, ta đã có sẵn một số điểm, sau đó ta dự đoán một số giá trị của $a$ và $b$, sử dụng điểm đó, ta vẽ một đường thẳng trên biểu đồ; sau khi làm điều đó, ta nhận ra rằng đường thẳng mới không chạm chính xác vào cả ba điểm dữ liệu mà ta đã có, vì vậy bây giờ ta tính toán khoảng cách giữa các điểm ban đầu và đường dự đoán. Minh họa bằng hình vẽ:

![mse](mse.png){: width="500"}_Mean Squared Error Representation_

Có thể tính toán bằng công thức như sau:

$$
\text{Cost function}=\frac{1}{2m}\sum_1^m (f(x^i)-y^i)^2 \tag{1}
$$

Thuật ngữ đầu tiên $\dfrac{1}{2m}$ là một số không đổi, trong đó $m$ là số lượng điểm dữ liệu, trong trường hợp này là 3. Thuật ngữ $f(x^i) $ có nghĩa là giá trị dự đoán của ML model cho giá trị cụ thể của $ i$, nói cách kháclà ta dự đoán bằng cách sử dụng phương trình $f(x)=ax + b$ và thuật ngữ $y^i$ có nghĩa là giá trị thực tế của điểm dữ liệu. Để dễ hình dung hơn, ta hãy thử làm ví dụ:

Với $(1, 1), (2, 2), (3, 3)$, ta có 

$$
\text{Cost function}=\dfrac{1}{2\times 3}(1-1 + 2-2 + 3-3)^2=0
$$

Với đường màu đỏ $(1, 1), (2, 3), (3, 6)$, ta có 

$$
\text{Cost function}=\dfrac{1}{2\times 3}((1-1)^2 + (3.33-3)^2 + (5.89-6)^2)=0.121
$$

## Thuật toán tối ưu tham số để giảm Cost function

Như ta thấy ở ví dụ trên, cost function bằng 0 với nghiệm đúng và khác 0 khi không tìm được nghiệm chính xác. Mục tiêu của mọi bài toán ML và deep learning là tìm các tham số sao cho cost function nhỏ nhất. Trong trường hợp này, ta muốn tìm được được 2 tham số $a,b$ sao cho hàm $f(x)=ax+b$ cho ra sai số nhỏ nhất.

## References

1.  <https://towardsdatascience.com/machine-leaning-cost-function-and-gradient-descend-75821535b2ef>
