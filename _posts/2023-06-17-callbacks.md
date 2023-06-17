---
title: Callbacks - Tăng cường kiểm soát và hiệu suất huấn luyện models
author: hoanglinh
categories: [Deep Learning, Machine Learning]
tags: [training tips]
math: true
---

Đó là một ngày đầy thách thức khi tôi đang tận hưởng cuộc sống của một anh developer thực thụ. Tôi đã dành hàng tuần để huấn luyện mô hình machine learning của mình, hy vọng sẽ tạo ra một kết quả ấn tượng. Nhưng ôi thôi, định mệnh đã chơi xỏ tôi.

Trong khi đang say mê với quá trình huấn luyện, một cúp điện đột ngột đã đánh bật toàn bộ hệ thống. Bầu không khí bị cắt đứt, đèn sáng tắt ngấm ngầm và mô hình của tôi... Đã mất! Tôi ngã lăn ra sàn, không chỉ vì tôi mất công làm việc suốt tháng ngày, mà còn vì tôi sẽ phải đối mặt với sự giận dữ của sếp.

Nhưng không phải lúc nào cũng là ngày xấu, phải không? Một thiên thần đã xuất hiện trong bóng tối! Một người đã tiếp cận tôi và giới thiệu về một kỹ thuật cứu cánh, đó chính là `callbacks()`. Ngay lập tức, tôi đã lắng nghe chăm chỉ và nhận ra rằng `callbacks()` là chìa khóa để vượt qua khủng hoảng này.

`callbacks()` không chỉ giúp tôi giải quyết vấn đề mất điện, mà còn là một trợ thủ đắc lực trong việc kiểm soát và tối ưu hóa quá trình huấn luyện mô hình. Từ việc kiểm soát độ đo, hàm mất mát, cho đến việc ngừng huấn luyện sớm, điều chỉnh tỷ lệ học và lưu trữ mô hình tốt nhất - `callbacks()` đơn giản là một siêu anh hùng đầy thực lực!

# Callbacks - Giải pháp cho quá trình training và monitoring

> Bằng cách sử dụng `callbacks()`, chúng ta có thể thực hiện các tác vụ quan trọng như ngừng huấn luyện sớm, điều chỉnh tỷ lệ học, lưu trữ mô hình tốt nhất và tăng cường quá trình tối ưu hóa. Với tác dụng và lợi ích của mình, callbacks là một công cụ không thể thiếu trong quá trình phát triển và huấn luyện mô hình machine learning.
{: .prompt-info}

1. **Theo dõi và ghi lại thông tin trong quá trình huấn luyện**: Callbacks cho phép chúng ta theo dõi và ghi lại các thông tin quan trọng như acc, hàm mất mát, trạng thái của mô hình sau mỗi epoch. Điều này giúp chúng ta kiểm tra và phân tích hiệu suất của mô hình trong quá trình huấn luyện.
2. **Early stopping**: cho phép chúng ta dừng quá trình huấn luyện khi các tiêu chí dừng được đáp ứng. Ví dụ, nếu acc của mô hình không cải thiện sau một số epoch, ta có thể dừng lại để tránh việc tiếp tục huấn luyện không cần thiết và tiết kiệm thời gian.
3. **Learning rate schedule**: theo lịch trình cụ thể trong quá trình huấn luyện. Điều này giúp tăng khả năng tìm ra điểm cực tiểu tối ưu của hàm mất mát.
4. **Lưu trữ mô hình tốt nhất**: Callbacks như ModelCheckpoint cho phép lưu trữ mô hình tốt nhất dựa trên độ đo nào đó, ví dụ như độ chính xác. Điều này giúp chúng ta không bỏ qua mô hình tốt nhất và có thể sử dụng nó sau này để dự đoán trên dữ liệu mới.
5. **Tự động tăng số lượng epoch**: Callbacks như `ReduceLROnPlateau` cho phép tự động điều chỉnh số lượng epoch và tỷ lệ học dựa trên hiệu suất của mô hình. Điều này giúp tối ưu hóa quá trình huấn luyện và tăng cường khả năng tìm ra mô hình tốt nhất.

## Theo dõi và ghi lại thông tin trong quá trình huấn luyện
Để theo dõi và phân tích lịch sử huấn luyện một cách dễ dàng, chúng ta có thể sử dụng `CSVLogger` và `TensorBoard` callbacks. `CSVLogger` ghi lại thông tin huấn luyện vào một tệp CSV, trong khi `TensorBoard` tạo ra các biểu đồ và đồ thị để hiển thị quá trình huấn luyện theo thời gian.

```python
csv_logger_callback = tf.keras.callbacks.CSVLogger('training.log')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
```

Sau khi đã có file `logs`, bạn có thể dùng câu lệnh này để xem trên `Tensorboard`

```python
tensorboard --logdir=path_to_your_logs
```

## Early stopping
Quá trình huấn luyện có thể mất rất nhiều thời gian và tài nguyên. Để tránh việc tiêu tốn thêm thời gian không cần thiết, chúng ta có thể sử dụng `EarlyStopping()` callback. Callback này sẽ ngừng quá trình huấn luyện nếu không có cải thiện đáng kể trong một khoảng thời gian quy định trên tập kiểm tra.

```python
early_stopping_callback = \
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',  # name of monitoring metrics
                                      min_delta=0,  # minimum improvement for every epoch
                                      patience=3,   # number of epochs to wait before stopping the training
                                      restore_best_weights=True,  # if True, restore to best weights if acc decrease
                                      verbose=0,  # print addition logs
                                      model='auto'  # trend of metrics (up, down) -> min, max, auto
                                      )
```

## Điều chỉnh learning rate (Learning rate schedule)
Learning rate là một tham số quan trọng trong quá trình huấn luyện mô hình. Để tối ưu hóa quá trình huấn luyện, chúng ta có thể sử dụng `LearningRateScheduler()` callback. Callback này cho phép điều chỉnh learning rate theo lịch trình tùy chỉnh, ví dụ: giảm learning rate sau mỗi epoch.

```python
def lr_schedule(epoch):
    learning_rate = 0.1
    if epoch > 10:
        learning_rate = 0.01
    return learning_rate

lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
```

## ModelCheckpoint - Tự động lưu trữ mô hình tốt nhất
Khi huấn luyện một mô hình, chúng ta thường quan tâm đến mô hình có độ chính xác tốt nhất trên tập kiểm tra. Để tự động lưu trữ mô hình tốt nhất, chúng ta có thể sử dụng `ModelCheckpoint()` callback. Callback này sẽ lưu trữ mô hình sau mỗi lần huấn luyện nếu nó có độ chính xác tốt hơn mô hình đã lưu trữ trước đó.

```python
checkpoint_callback = \
    tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', 
                                        monitor='val_accuracy',  
                                        save_best_only=True,  # if True, best model không bị ghi đè lên
                                        save_weights_only=False,  # chỉ lưu weights hay cả model 
                                        save_freq='epoch'  # lưu sau mỗi epoch
                                        )

```

## ReduceLROnPlateau - Thay đổi learning_rate khi model không cải thiện nữa
Sau một số epoch mà model không có dấu hiệu tăng accuracy nữa, chúng ta có thể sử dụng phương pháp này để giảm `learning_rate` (dựa theo metric chứ không phải epoch) đi để xem model có tốt hơn không.

```python
tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                     factor=0.1,  # factor to decrease learning rate = old_lr * factor
                                     patience=10,  
                                     verbose=0, 
                                     mode='auto',    
                                     min_delta=0.0001, 
                                     cooldown=0,  # number of epochs to wait before restarting the monitoring 
                                     min_lr=0,  # min bound for learning rate
                                     **kwargs)
```

# Sử dụng thực tế với Callback
Callbacks đã giúp bạn trở lại từ tình huống khi quá trình huấn luyện bị gián đoạn. Với các Callbacks như `EarlyStopping`, `ModelCheckpoint`, `LearningRateScheduler`, `CSVLogger` và `TensorBoard`, bạn đã có trong tay các công cụ mạnh mẽ để tối ưu hóa quá trình huấn luyện mô hình của mình. Bằng cách sử dụng Callbacks, bạn có thể tiết kiệm thời gian, tăng hiệu suất và đạt được kết quả tốt hơn cho mô hình machine learning / deep learning của bạn.

Bạn có thể sử dụng callback trong quá trình `fit` model:
```python
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          validation_data=(X_val, y_val), 
          epochs=20, 
          callbacks=[checkpoint_callback, early_stopping_callback, lr_scheduler_callback, csv_logger_callback, tensorboard_callback])
```

# Recommended resources for further learning
Here are some recommended books for delving into deep learning from scratch:
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://amzn.to/3YZeOAk) by Aurélien Géron
- [Deep Learning from Scratch: Building with Python from First Principles](https://amzn.to/40gyjFQ) by Seth Weidman
- [Data Science from Scratch: First Principles with Python](https://amzn.to/40ep3T7) by Joel Grus, a research engineer at the Allen Institute for Artificial Intelligence

# References
1. [A Guide to TensorFlow Callbacks By Keshav Aggarwal](https://blog.paperspace.com/tensorflow-callbacks/)