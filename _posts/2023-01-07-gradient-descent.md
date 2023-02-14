---
title: Thuật toán Gradient Descent (GD) với Python
author: hoanglinh
categories: [Deep Learning]
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
\theta^{t+1} =\theta - \eta  ∇_\theta \tag{3}
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

Để tìm được optimal weight cho các features, thuật toán GD sẽ tính gradient của hàm loss đối với mỗi model parameters $\theta_j$. Hay nói cách khác, ta cần biết loss thay đổi bao nhiêu nếu ta thay đổi giá trị của $\theta_j$ với một lượng nhất định, gọi là đạo hàm riêng (partial derivative). Để tính đạo hàm riêng, ta có thể sử dụng công thức:

$$
\frac{\partial}{\partial\theta_j}MSE(\mathbf{\theta})  = \frac{2}{m}\sum_{i=1}^m(\mathbf{\theta}^Tx^{(i)} - y^{(i)})x^{(i)}_j \tag{4}
$$

> Sở dĩ ta gọi công thức trên là batch GD vì nó tính toán dựa trên toàn bộ data. Việc sử dụng tất cả data để tính một lần có thể gây nên hiện tượng training rất lâu và khối lượng tính toán lớn. Nhưng vẫn rất nhanh khi so sánh với phương pháp tìm nghiệm thông thường, đặc biệt là khi số  lượng features tăng lên hàng trăm hoặc thậm chí hàng ngàn. 
> {: .prompt-info}

Mean squared error có thể tính theo code dưới đây:

```python
def mean_squared_error(y_true, y_predicted):
	
    # Calculating the loss or cost
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost
```

Khi chúng ta có vector độc dốc và vị trí hiện tại, chúng ta chỉ cần đi ngược lại với vector độ dốc. Nghĩa là ta phải trừ θ đi 1 giá trị là $∇_\theta MSE(\theta)$. Lúc này ta sẽ sử dụng tham số learning rate $\eta$ để xác định giá trị của bước xuống dốc bằng cách nhân vào.
$$
\theta^{(\text{next step})} =\theta - \eta  ∇_\theta MSE(\theta) \tag{5}
$$
Vậy là công thức về cơ bản đã đủ, ta hãy cùng triển khai với Python, trước tiên là dataset để validate thuật toán GD. ta sẽ random ra 100 data gồm 2 features:

```python
X = np.random.randn(100, 1)
y = 4 + 3*X + np.random.randn(100, 1)

plt.scatter(X, y)
plt.show()
```

![gd-data](gd-data.png)

Tiếp theo, ta sẽ code GD dựa vào các công thức bên trên

```python
def gradient_descent(x, y, iterations = 1000, learning_rate = 0.01,
    stopping_threshold = 1e-6):

    # Initializing weight, bias, learning rate and iterations
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))

    costs = []
    weights = []
    previous_cost = None

    # Estimation of optimal parameters
    for i in range(iterations):

        # Making predictions
        y_predicted = (current_weight * x) + current_bias

        # Calculating the current cost
        current_cost = mean_squared_error(y, y_predicted)

        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break

        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)

        # Calculating the gradients
        weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        bias_derivative = -(2/n) * sum(y-y_predicted)

        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
```

Sau khi đã code xong GD, ta có thể kiểm tra đối với data đã tạo bên trên

```python
# Estimating weight and bias using gradient descent
estimated_weight, estimated_bias = gradient_descent(X, Y, iterations=2000)
print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

# Making predictions using estimated parameters
Y_pred = estimated_weight*X + estimated_bias

# Plotting the regression line
plt.figure(figsize = (8,6))
plt.scatter(X, Y, marker='o', color='red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)],
         color='blue', markerfacecolor='red', markersize=10,linestyle='dashed')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

![gd-loss-update](gd-cost-update.png)

Kết quả của 2 tham số weight và bias sau khi sử dụng GD là 2.84 và 4.20, khá gần với giá trị giả sử ban đầu. Rất khó để ta có thể tìm được giá trị thực như giả sử vì các data còn bị ảnh hưởng bởi nhiễu bên ngoài. Trong thực tế, việc xác định được xu hướng đúng và giá trị loss thấp vừa phải đã là thành công đối với các ML model.

![fit-line](fit-line.png)

Ngoài ra, còn một tham số nữa cũng rất quan trọng như đã đề cập ở phần trên, đó là `learning_rate`. Hình dưới minh họa quá trình optimize của GD khi ta đặt các giá trị `learning_rate` khác nhau. Có thể thấy, nếu ta chọn được các tham số phù hợp, ta có thể tìm được weight và bias gần với giá trị đúng nhất. Ngược lại, nếu tham số ta chọn không hiệu quả có thể làm model không thể hoặc rất lâu mới tìm được tham số optimal cho weight và bias.

![gd-update-0.01](gd-update-0.01.gif)

![gd-update-0.5](gd-update-0.5.gif)

Vậy thì câu hỏi là làm thế nào để tìm được các tham số phù hợp? Grid search có thể là một giải pháp nhưng phương pháp này tốn nhiều thời gian vì nó sẽ phải thử từng cặp giá trị của các tham số. Đây cũng là một hướng nghiên cứu, các bạn có thể tìm đọc thêm lại đây: [Hyperparameters Optimization](https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d).

> Khi cost function is convex và độ dốc của nó không thay đổi đột ngột (như trường hợp của hàm chi phí MSE), Batch Gradient Descent với tốc độ học (`learning_rate`) cố định cuối cùng sẽ hội tụ về giải pháp tối ưu, nhưng bạn có thể phải đợi một lúc: nó có thể lặp lại $O(1/\epsilon)$ để đạt được mức tối ưu trong phạm vi ε, tùy thuộc vào hình dạng của hàm chi phí. Nếu bạn chia dung sai cho 10 để có giải pháp chính xác hơn, thì thuật toán có thể phải chạy lâu hơn khoảng 10 lần.

## Referenes

1. https://ndquy.github.io/posts/gradient-descent-2/
1. https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d