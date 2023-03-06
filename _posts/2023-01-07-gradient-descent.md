---
title: Thuật toán Gradient Descent (GD) với Python
author: hoanglinh
categories: [Optimization Algorithms]
tags: []
math: true
img_path: posts_media/2023-01-07-gradient-descent/
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

Chắc hản bạn đọc có thể thắc mắc làm sao để có được công thức (4). Như đã đề cập ở trên, thuật toán GD sẽ tính đạo hàm riêng của hàm loss, công thức (1), đối với từng biến, ở đây là weight `w` và bias `b`, số 2 ngoài tổng là vì ta lấy đạo hàm của hàm số loss bậc 2, `m` là số lượng data, chỉ tiết được thể hiện như code `gradient_descent()` bên dưới. 

> Sở dĩ ta gọi công thức trên là batch GD vì nó tính toán dựa trên toàn bộ data. Việc sử dụng tất cả data để tính một lần có thể gây nên hiện tượng training rất lâu và khối lượng tính toán lớn. Nhưng vẫn rất nhanh khi so sánh với phương pháp tìm nghiệm thông thường, đặc biệt là khi số  lượng features tăng lên hàng trăm hoặc thậm chí hàng ngàn. 
{: .prompt-tip }

Mean squared error có thể tính theo code dưới đây:

```python
def mean_squared_error(y_true, y_predicted):
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
y = 3 + 4*X + np.random.randn(100, 1)

plt.scatter(X, y)
plt.show()
```

![gd-data](gd-data.png)

Tiếp theo, ta sẽ code GD dựa vào các công thức bên trên

```python
def batch_gradient_descent(X, y, weight, bias, learning_rate=0.01, num_iterations=200):
    training_size = X.shape[0]

    for idx in range(num_iterations):
        weight_derivative = -(2 / training_size) * sum(X * (y - (weight * X + bias)))
        bias_derivative = -(2 / training_size) * sum(y - (weight * X + bias))

        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

        loss = mean_squared_error(y, weight * X + bias)
        print(f'Loss at iteration {idx}: {loss}')

    return weight, bias
```

Sau khi đã code xong GD, ta có thể kiểm tra đối với data đã tạo bên trên

```python
weight = np.random.random()
bias = np.random.random()

learning_rate = 0.1
num_epochs = 500
weight, bias = batch_gradient_descent(X, y, weight, bias, learning_rate, num_epochs)
print(f"Estimated Weight: {weight}\nEstimated Bias: {bias}")

# Making predictions using estimated parameters
Y_pred = weight*X + bias

# Plotting the regression line
plt.scatter(X, y, marker='o', color='red')
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)],
         color='blue', markerfacecolor='red', markersize=10,linestyle='dashed')
plt.xlabel("X")
plt.ylabel("y")
plt.title('Fitting line with trained weight and bias')
plt.show()
```

![gd-loss-update](gd-cost-update.png)

Kết quả của 2 tham số weight và bias sau khi sử dụng GD là 2.84 và 4.20, khá gần với giá trị giả sử ban đầu. Rất khó để ta có thể tìm được giá trị thực như giả sử vì các data còn bị ảnh hưởng bởi nhiễu bên ngoài. Trong thực tế, việc xác định được xu hướng đúng và giá trị loss thấp vừa phải đã là thành công đối với các ML model.

![fit-line](fit-line.png)

Ngoài ra, còn một tham số nữa cũng rất quan trọng như đã đề cập ở phần trên, đó là `learning_rate`. Nếu ta chọn được các tham số phù hợp, ta có thể tìm được weight và bias gần với giá trị đúng nhất. Ngược lại, nếu tham số ta chọn không hiệu quả có thể làm model không thể hoặc rất lâu mới tìm được tham số optimal cho weight và bias. Vậy thì câu hỏi là làm thế nào để tìm được các tham số phù hợp? Grid search có thể là một giải pháp nhưng phương pháp này tốn nhiều thời gian vì nó sẽ phải thử từng cặp giá trị của các tham số. Đây cũng là một hướng nghiên cứu, các bạn có thể tìm đọc thêm lại đây: [Hyperparameters Optimization](https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d).

> Khi cost function is convex và độ dốc của nó không thay đổi đột ngột (như trường hợp của hàm chi phí MSE), Batch Gradient Descent với tốc độ học (`learning_rate`) cố định cuối cùng sẽ hội tụ về giải pháp tối ưu, nhưng bạn có thể phải đợi một lúc: nó có thể lặp lại $O(1/\epsilon)$ để đạt được mức tối ưu trong phạm vi ε, tùy thuộc vào hình dạng của hàm chi phí. Nếu bạn chia dung sai cho 10 để có giải pháp chính xác hơn, thì thuật toán có thể phải chạy lâu hơn khoảng 10 lần.
{: .prompt-info}

## Stochastic Gradient Descent

Vấn đề chủ yếu của phương pháp Batch Gradient Descent là thời gian training lâu vì nó phải thực hiện tính toán và tối ưu weight và bias dựa trên cả dataset. Ngược lại, phương pháp Stochastic GD lựa chọn bất kỳ một data trong tập training và tính toán gradient. Hiển nhiên tốc độ tính toán sẽ nhanh hơn nhiều vì chỉ cần tính trên một sample ở mỗi iteration. Đồng thời nó cũng giúp ta train được với các tập training lớn vì ta không cần load hết data vào memory mỗi iteration, điều mà có thể là nguyên nhân gây lỗi bộ nhớ khi sử dụng Batch GD.

Tuy nhiên, vì tính chất stochastic, giá trị loss thường không giảm dần đều như batch GD. Điều này có thể giải thích vì mỗi sample có các features và pattern khác nhau, vì vậy kết quả tính gradients và update weight và bias cũng khác và từ đó làm loss lúc tăng lúc giảm. Mặc dù xu thế chung của hàm loss cũng là giảm nhưng nó sẽ có biến động cao thấp ở mỗi vòng lặp. Sau một số  vòng lặp, giá trị loss có thể giảm xuống thấp và chấp nhận được nhưng vẫn có khả năng biến động nhỏ xung quanh điểm thực sự optimal. 

Cũng chính vì tính chất lên xuống, stochastic GD có khả năng tìm được global optima thay vì bị mắc kẹt ở local optima. Lợi dụng tính chất này, ta có thể viết lại GD chút để thuật toán vừa có khả năng tìm được global optima và hội tụ ổn định tại điểm đó. Chúng ta có thể giảm dần `learning_rate`, trong đó giá trị cao làm tăng khoảng cách tìm kiếm (khám phá) và giá trị nhỏ tập trung vào điểm quan trọng. Câu hỏi đặt ra là giảm thế nào cho hợp lý, giảm nhanh thì dễ bị local optimal, giảm chậm thì thời gian training lâu. Thử cách đơn giản như:

```python
def learning_rate_schedule(num_epochs, epoch, sample):
    return (num_epochs - epoch) / sample
```

Sau khi đã có hàm giảm dần `learning_rate`, ta tiến hành train với chỉ 20 data mỗi epoch và cho chạy 20 epochs:

```python
def stochastic_gradient_descent(X, y, weight, bias, num_epochs=100, num_train_sample=30):
    training_size = X.shape[0]

    for epoch in range(1, num_epochs):
        train_sample_idx = np.random.randint(low=0, high=training_size, size=num_train_sample)
        train_sample_data = np.take(X, train_sample_idx, axis=0)
        train_sample_label = np.take(y, train_sample_idx, axis=0)

        weight_derivative = -(2 / training_size) * sum(
            train_sample_data * (train_sample_label - (weight * train_sample_data + bias)))
        bias_derivative = -(2 / training_size) * sum(train_sample_label - (weight * train_sample_data + bias))

        # calculate learning rate
        learning_rate = learning_rate_schedule(epoch)

        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

    return weight, bias
```

```python
num_epochs = 50
weight, bias = stochastic_gradient_descent(X, y, weight, bias, num_epochs, num_train_sample=30)
```

Wow, kết quả cho ra `weight = 3.938` và `bias = 2.872`, trong đó `loss = 1.008`. Ta có thể thấy chỉ với số lượng nhỏ data mà thuật toán cũng có thể tìm được solution khá tốt. 

## Mini-batch Gradient Descent

Phương pháp cuối cùng thuộc GD là mini-batch GD, là sự kết hợp ưu và nhược điểm giữa batch và stochastic GD. Nói cách khác, thay vì train cả tập data hay chỉ dùng 1 sample để tính toán, ta có thể tính gradients trên một tập nhỏ data. 

> Ưu điểm chính của Mini-batch GD so với Stochastic GD là ta có thể tăng performance từ việc tối ưu hóa phần cứng cho các tính toán ma trận, đặc biệt là khi sử dụng GPU để tính toán song song. Điều này có thể thực hiện bằng cách sử dụng thư viện `numpy`.
{: .prompt-info}

```python
def mini_batch_gradient_descent(X, y, weight, bias, num_epochs=100, num_train_sample=30):
    training_size = X.shape[0]

    for epoch in range(1, num_epochs):
        train_sample_idx = np.random.randint(low=0, high=training_size, size=num_train_sample)
        train_sample_data = np.take(X, train_sample_idx, axis=0)
        train_sample_label = np.take(y, train_sample_idx, axis=0)

        weight_derivative = -(2 / training_size) * sum(train_sample_data * (train_sample_label - \
                                                                            (weight * train_sample_data + bias)))
        bias_derivative = -(2 / training_size) * sum(train_sample_label - (weight * train_sample_data + bias))

        weight -= learning_rate * weight_derivative
        bias -= learning_rate * bias_derivative

    return weight, bias

```

## So sánh và Kết luận

Hình dưới cho thấy các giá trị `weight` được thực hiện bởi 3 thuật toán Gradient Descent trong quá trình training. Tất cả đều kết thúc ở mức gần điểm optima, nhưng đường đi của batch GD dừng lại ở mức optima tốt nhất, trong khi cả Stochastic GD và mini-batch GD tiếp tục đi lòng vòng. Tuy nhiên, batch GD cần rất nhiều thời gian, trong khi đó Stochastic GD và mini-batch GD cũng sẽ đạt được optima nếu ta tìm được các hyper-parameters tốt.

![gd-comparison](gd-comparison.png)_Quá trình update của Batch, Stochastic và Mini-batch GD. Source: https://www.analyticsvidhya.com_

## Referenes

1. [https://ndquy.github.io/posts/gradient-descent-2/](https://ndquy.github.io/posts/gradient-descent-2/)
1. [https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d](https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d)
1. [https://oyane806.github.io/dl-in-minutes/](https://oyane806.github.io/dl-in-minutes/)