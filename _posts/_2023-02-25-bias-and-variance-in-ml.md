---
title: Khái niệm Bias và Variance trong Machine Learning
author: hoanglinh
categories: [Fundamental Concepts]
tags: []
math: true
img_path: posts_media/2023-02-25-bias-and-variance-in-ml/
---

Chắc hẳn trong quá trình xây dựng mô hình bạn đã từng đối mặt với vấn đề mô hình dự báo tốt trên tập huấn luyện nhưng không dự báo tốt trên tập kiểm tra. Trước khi đọc bài viết này, bạn không hiểu nguyên nhân từ đâu và khắc phục như thế nào. Bài viết này sẽ cung cấp cho bạn các kiến thức liên quan tới lỗi mô hình, cách phòng tránh cũng như làm thế nào để khắc phục chúng.

## Thế nào là Errors trong Machine Learning?

Trong Machine Learning, error là thước đo mức độ chính xác của một thuật toán có thể đưa ra dự đoán cho một tập dữ liệu chưa biết trước đó (unseen data). Trên cơ sở các lỗi này, ta có thể biết mô hình học máy được chọn có thể hoạt động tốt nhất trên tập dữ liệu mới hay không. Thông thường, có thể phân loại thành 2 loại lỗi chính, lỗi có thể sửa được và lỗi không thể sửa được. 

Đối với các lỗi có thể sửa được, ví dụ như sai số hoặc feature thu được từ model chưa đủ để phân biệt với các class khác, chúng ta có thể tăng mức độ phức tạp, thêm layer, etc. để cải thiện độ chính xác của mô hình. Những lỗi như vậy có thể được phân loại thành sai lệch và phương sai, chi tiết đề cập ở phần sau. Ngược lại, lỗi không thể sửa được có thể đến từ vấn đề data không sạch hoặc outliers, điều này rất hay gặp trong các data thực tế.

![ml-error](ml-error.png)_Source: (Learning a Multiview Weighted Majority Vote Classifier: Using PAC-Bayesian Theory and Boosting)_

## Độ chệch (bias) và phương sai (variance) là gì?

Nói chung, một mô hình Machine Learning sẽ phân tích dữ liệu, tìm các features và patterns trong data và đưa ra dự đoán. Trong quá trình training, model học các features này trong tập dữ liệu train và áp dụng chúng để kiểm tra dữ liệu để dự đoán. Trong khi đưa ra dự đoán, sự khác biệt xảy ra giữa các giá trị dự đoán do mô hình tạo ra và giá trị thực tế (ground truth) được gọi là lỗi sai lệch (bias error). Nó có thể được hiểu như là khả năng của các thuật toán Machine Learning như Hồi quy tuyến tính (linear regression) học được quan hệ thực sự giữa các điểm dữ liệu. Mỗi thuật toán bắt đầu với một lượng bias nhất định vì sai lệch xảy ra từ các giả định trong mô hình, sau quá trình update, bias sẽ giảm dần.

Một model Machine Learning có thể gặp phải tình trạng **low hoặc high bias**. 

- **High bias**: hàm dự đoán $f(x)$ của model cho ra giá trị dự báo khác ground truth nhiều. Thông thường những mô hình **quá đơn giản** được huấn luyện trên những bộ dữ liệu **lớn** sẽ dẫn tới độ chệch lớn. Nguyên nhân của bị chệch thường là do mô hình **quá đơn giản** trong khi dữ liệu có mối quan hệ phức tạp hơn và thậm chí nằm ngoài khả năng biểu diễn của mô hình.
- **Low bias**: hàm dự đoán $f(x)$ phức tạp hơn, nó cố để fit training data sao cho sai số giữa predict và actual thấp nhất. Tuy nhiên, trong một số tình huống, một mô hình quá phức tạp cũng có khả năng xảy ra hiện tượng phương sai cao (high variance).

> Nói chung, thuật toán tuyến tính (eg. linear regression) thường có bias cao, vì nó đơn giản và học nhanh. Thuật toán càng đơn giản thì khả năng xảy ra sai lệch càng cao. Trong khi đó thuật toán phi tuyến tính (polynomial) thường cho ra bias thấp hơn.
{: .prompt-tip }

![low-high-bias](low-high-bias.png)_High vs low bias - Source: <https://buggyprogrammer.com/bias-vs-variance-tradeoff/>_

**Phương sai (variance)** có thể hiểu là hiện tượng model dự báo có độ dao động lớn nhưng lại thiếu tính tổng quát về xu hướng hay đặc tính của tổng thể data. Những lớp mô hình **phức tạp** được huấn luyện trên tập huấn luyện **nhỏ và đơn giản** thường xảy ra hiện tượng phương sai cao (high variance) và dẫn tới việc ***học vẹt*** thông qua bắt chước dữ liệu hơn là học qui luật tổng quát. 

Khi mô hình có độ chệch lớn hoặc phương sai lớn đều ảnh hưởng tới hiệu suất dự báo. Vì vậy chúng ta cần giảm thiểu chúng để tăng cường sức mạnh cho mô hình. Thực ra khái niệm high bias và high variance khá trìu tượng và nhiều lúc dùng nhầm lẫn giữa thống kê và machine learning. Nên khái niệm hay được dùng hơn là **underfitting** và **overfitting**.

Ví dụ khi luyện thi đại học, nếu bạn chỉ luyện khoảng 1-2 đề trước khi thi thì bạn sẽ bị **underfitting** vì bạn chưa hiểu hết cấu trúc, nội dung của đề thi. Tuy nhiên nếu bạn chỉ luyện kĩ 50 đề thầy cô giáo bạn soạn và đưa cho thì khả năng bạn sẽ bị **overfitting** với các đề mà thầy cô giáo các bạn soạn mà khi thi đại học có thể điểm số của các bạn vẫn tệ.

![tradeoff](Bias-Variance-Tradeoff.png)_Source: <https://www.ml-science.com/bias-variance-tradeoff>_

## Đánh giá và Giải pháp

Có 2 thông số thường được sử dụng để đánh giá bias and variance của mô hình là training set error và validation set error. Ví dụ error (1-accuracy) trong logistic regression. Ta mong muốn model là low bias và low variance.

| Train set error |      **1%**       |    15%    |           **15%**           |         0.5%          |
| :-------------- | :---------------: | :-------: | :-------------------------: | :-------------------: |
| Val set error   |      **11%**      |    16%    |           **30%**           |          1%           |
|                 | **High variance** | High bias | **High bias High variance** | Low bias Low variance |

**Giải quyết high bias (underfitting)**: 

- Ta cần tăng độ phức tạp của model, tăng số lượng hidden layer và số node trong mỗi hidden layer.
- Dùng nhiều epochs hơn để train model.

**Giải quyết high variance (overfitting):**

- Thu thập thêm dữ liệu hoặc dùng [data augmentation](https://nttuan8.com/bai-9-transfer-learning-va-data-augmentation/#Data_augmentation)
- Dùng regularization như: L1, L2, **droupout**

## References

1. <https://nttuan8.com/bai-10-cac-ky-thuat-co-ban-trong-deep-learning/#Bias_va_variance>
