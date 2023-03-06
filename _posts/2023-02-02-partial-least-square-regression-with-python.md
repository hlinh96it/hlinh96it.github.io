---
title: Thuật toán Partial Least Squares (PLS) với Python (Phần 1)
author: hoanglinh
categories: [Machine Learning]
tags: [dimensional reduction, regression]
math: true
img_path: posts_media/2023-02-02-partial-least-square-regression-with-python/
---

Trong bài viết này, chúng ta sẽ cùng nhau tìm hiểu về Partial Least Squares (PLS). Đúng như tên gọi của nó, PLS có liên quan đến Ordinary Least Squares, nó là một phương pháp toán học cơ bản để tìm ra đường fitting linear regression.

Dành cho các bạn mới làm quen với Linear regression, các bạn có thể tìm các bài viết liên quan: [thuật toán machine learning linear regression](https://machinelearningcoban.com/2016/12/28/linearregression/) và các [giả định đối với data của linear regression](https://www.notion.so/deba871a26d34c67bac023a447fab221).

## Partial Least Squares đối với hiện tượng đa cộng tuyến (multi-collinearity)

Hiểu về cơ bản, hiện tượng đa cộng tuyến trong linear regression là hiện tượng các biến phụ thuộc tuyến tính lẫn nhau, thể hiện dưới hàm số và vi phạm giả định số 5 của mô hình linear regression. Bạn đọc quan tâm có thể tham khảo bài viết về [hệ quả của đa cộng tuyến với mô hình linear regression](https://www.notion.so/6cb85c40ecdf4a3abbd01f634f1c3508).

![da-cong-tuyen](Da-cong-tuyen.png)_Nguồn: Xử lý số liệu SPSS_

Mục tiêu của linear regression là mô phỏng lại mối liên hệ phụ thuộc giữa target và một hoặc nhiều variables (hay còn gọi là inputs). Mô hình OLS có thể cho ra kết quả tốt nếu các variable không vi phạm [các giả định](https://www.notion.so/deba871a26d34c67bac023a447fab221). Tuy nhiên, thực tế là các inputs có thể có mối liên hệ với nhau và từ đó, kết quả của OLS không còn đáng tin cậy nữa.

PLS được sinh ra để giải quyết vấn đề đa cộng tuyến trong linear regression bằng cách giảm kích thước của các biến tương quan quan đến nhau và mô hình thông tin cơ bản của các biến đó.

## PLS đối với bài toán multi-variate outcome

Một điểm đặc biệt nữa của thuật toán PLS là khả năng model được bài toán với nhiều outcomes trong khi đó các thuật toán thống kê hoặc machine learning không thể làm được điều đó trực tiếp.

Mặc dù ta có thể xây model cho mỗi biến, tuy nhiên mọi yếu tố cần được giữ trong model đặc biệt là với các bài toán mà các biến có sự tương quan với nhau. Hơn nữa, việc giải thích mô hình multi-variate outcomes sẽ khác với việc giải thích nhiều mô hình univariate.

## So sánh PLS với các phương pháp khác

Trước khi đi vào chi tiết phương pháp PLS, hãy cùng tìm hiểu xem lý do tại sao và trong trường hợp nào nên dùng PLS. So sánh với với các thuật toán khác để giải quyết vấn đề (1) có nhiều biến phụ thuộc (multiple outputs) và (2) có nhiều biến độc lập (inputs) tương quan với nhau

|                  Phương pháp                  | Trường hợp sử dụng                                           | Giới hạn                                                     |
| :-------------------------------------------: | :----------------------------------------------------------- | ------------------------------------------------------------ |
| Multi-variate<br /> multiple <br />regression | Nhiều inputs và nhiều outputs regression                     | Không có khả năng xử lý hiện tượng đa cộng tuyến             |
|  Principle <br />component <br />regression   | Các inputs có liên quan đến nhau. <br />Sử dụng PCA trước rồi cho vào OLS model. <br />PCR tập trung vào giảm kích thước inputs bằng cách tập trung vào phương sai (variance) | Không phù hợp với bài toán tìm mối liên hệ phụ thuộc giữa output và inputs |
|  Canonical <br />correlation<br /> analysis   | Tìm mối tương quan (correlation) giữa 2 datasets<br />Thực hiện bằng cách giảm kích thước của cả 2 data <br />và tìm ra cặp component nào có độ tương quan cao nhất | Tập trung vào so sánh mức độ tương quan thay vì hiệp phương sai như PLS |

## Partial Least Squares models

Trước tiên, để tránh nhầm lẫn với các định nghĩa trong các nghiên cứu khác, chúng ta cần làm rõ các định nghĩa hoặc khái niệm.

- **PLS regression**: cũng là một họ của phương pháp PLS nhưng tập trung vào bài toán mà output là số
- **PLS discriminant analysis (PLS-DA)**: tập trung vào bài toán với output là categories (bài toán phân loại)
- **PLS1 và PLS2**: model với chỉ MỘT output và NHIỀU outputs
- **SIMPLS vs NIPALS**: là 2 phương pháp để triển khai PLS, trong đó SIMPLS được biết đến với khả năng tính toán nhanh hơn và đơn giản hơn so với người tiền nhiệm là NIPALS.
- **KernelPLS:** vì PLS là biến thể của phương pháp Linear regression nên nó bản chất không có khả năng làm việc với non-linear problems → KernelPLS sinh ra để giải quyết vấn đề đó
- **OPLS (orthogonal projects to latent structures)**: là một bản nâng cấp của PLS, OPLS dễ giải thích hơn. Nếu PLS chỉ phân chia variability thành systemic và noise, thì OPLS tiến thêm một bước bằng cách chia systemic variability thành predictive and orthogonal variability.

## Partial Least Squares Regression Example với Python

Trước tiên, ta sẽ làm việc với continuous data, đối với categorical data sẽ được đề cập ở phần sau.

Trong `Python`, ta có thể import `meats` data từ

```python
import pandas as pd
import boto

# import the csv file directly from an s3 bucket
data = pd.read_csv('<https://raw.githubusercontent.com/hlinh96it/implement_algorithm/main/Regression_Algorithms/meats.csv>')
data = data.drop('Unnamed: 0', axis = 1)
data
```

![meats-data](meats-data.png)

Chúng ta có thể chia data thành 3 phần là train, validation và test bằng code dưới đây. Các bạn cũng có thể sử dụng thư viện `train_test_split()` trong `sklearn`

```python
# Split data in train, val, test
meats_train = data.iloc[0:130,:]
meats_val = data.iloc[130:175,:]
meats_test = data.iloc[175:215,:]

# Split the columns in X and Y
X_colnames = data.columns[:-3]
Y_colnames = data.columns[-3:]

# Split each train, val and test into two arrays
X_train = meats_train[X_colnames].values
Y_train = meats_train[Y_colnames].values

X_val = meats_val[X_colnames].values
Y_val = meats_val[Y_colnames].values

X_test = meats_test[X_colnames].values
Y_test = meats_test[Y_colnames].values
```

Sau khi đã có data, ta tiến hành train và validate moadel bằng thư viện `sklearn`

```python
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
water, fat, protein = [], [], []
for n_comp in range(1, 101):
    my_plsr = PLSRegression(n_components=n_comp, scale=True)
    my_plsr.fit(X_train, y_train)
    preds = my_plsr.predict(X_val)

    water.append(np.sqrt(mean_squared_error(y_val['water'].values, preds[:, 0])))
    fat.append(np.sqrt(mean_squared_error(y_val['fat'], preds[:, 1])))
    protein.append(np.sqrt(mean_squared_error(y_val['protein'], preds[:, 2])))

fig, ax = plt.subplots(nrows=1, ncols=3, sharex='all', sharey='all', figsize=(16, 4))
ax[0].plot(range(1, 101), water)
ax[1].plot(range(1, 101), fat)
ax[2].plot(range(1, 101), protein)

plt.suptitle('Partial Least Squares — plot for the number of components', fontsize=18)
fig.supxlabel('Number of components', fontsize=15)
fig.supylabel('RMSE value', fontsize=15)
plt.tight_layout()
plt.show()
```

![output.png](pls_comps.png)

Từ hình vẽ trên, ta có thể thấy số lượng `component` cho ra kết quả `RMSE` nhỏ nhất nằm trong khoảng 20 → 30. Ta cũng có thể visualize sự khác nhau khi số lượng components thay đổi đối với kết quả. Vì chúng ta có tổng cộng là 100 inputs nên ta có tổng là 100 hệ số cho model regression.

Hiển nhiên là, khi số lượng components tăng lên, độ phức tạp của model cũng tăng theo. Vậy làm thế nào để biết được bao nhiêu `component` là tối ưu? Ta sẽ sử dụng phương pháp `Grid search` và tính giá trị `r-square`. Bạn đọc có thể tìm hiểu thêm về `r-square` tại [bài viết này](https://www.notion.so/37daf433b45347338cc5afbe990bbe7a).

```python
from sklearn.metrics import r2_score

best_r2 = 0
best_ncmop = 0

for n_comp in range(1, 101):
    my_plsr = PLSRegression(n_components=n_comp, scale=True)
    my_plsr.fit(X_train, y_train)
    preds = my_plsr.predict(X_val)

    r2 = r2_score(preds, y_val)
    if r2 > best_r2:
        best_r2 = r2
        best_ncomp = n_comp

print(best_r2, best_ncmop)
```

`r-square` tốt nhất mà chúng ta có thể đạt được là điểm `r-square` là 0,943 với giá trị `ncomp` là 16. Để xác thực lần cuối mô hình của chúng ta, hãy xác minh xem chúng ta có đạt được điểm tương đương trên tập dữ liệu thử nghiệm hay không:

```python
best_model = PLSRegression(n_components=best_ncmop, scale=True)
best_model.fit(X_train, Y_train)
test_preds = best_model.predict(X_test)
print(r2_score(Y_test, test_preds))
```

Điểm `r-square` thu được là 0,95628, cao hơn một chút so với kết quả trên tập `validation`. Chúng ta có thể tự tin rằng mô hình không bị overfitting và chúng ta đã tìm thấy số lượng `components` phù hợp để tạo nên một mô hình hoạt động hiệu quả.

> Nếu sai số này có thể chấp nhận được cho mục đích thử nghiệm thịt, thì chúng tôi có thể tự tin thay thế phép đo nước, chất béo và protein thủ công bằng phép đo hóa học tự động kết hợp với Công cụ hồi quy PLS này. Hồi quy PLS sau đó sẽ phục vụ để chuyển đổi các phép đo hóa học thành ước tính hàm lượng nước, chất béo và protein.
{: .prompt-info}

