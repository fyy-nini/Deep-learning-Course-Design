import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# 加载数据
train = pd.read_csv('downloads/136244/train.csv')  # 假设训练数据存储在 'train.csv' 文件中
test = pd.read_csv('downloads/136244/test.csv')    # 假设测试数据存储在 'test.csv' 文件中

# 特征选择（假设所有列都可以作为特征）
features = [col for col in train.columns if col != 'subscribe']
X = train[features]
y = train['subscribe']

# 处理字符串类型的特征
X = pd.get_dummies(X)
test = pd.get_dummies(test)

# 确保训练集和测试集具有相同的特征列
missing_cols_in_test = set(X.columns) - set(test.columns)
for c in missing_cols_in_test:
    test[c] = 0

missing_cols_in_train = set(test.columns) - set(X.columns)
for c in missing_cols_in_train:
    X[c] = 0

# 确保特征顺序一致
X = X[X.columns.sort_values()]
test = test[test.columns.sort_values()]

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 标签编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)

# 将标签转换为one-hot编码
num_classes = len(label_encoder.classes_)
y_train_onehot = to_categorical(y_train_encoded, num_classes=num_classes)
y_val_onehot = to_categorical(y_val_encoded, num_classes=num_classes)

# 构建神经网络模型
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train_scaled, y_train_onehot, epochs=100, batch_size=32, validation_data=(X_val_scaled, y_val_onehot), verbose=1)

# 评估模型
loss, accuracy = model.evaluate(X_val_scaled, y_val_onehot)
print(f'Validation Accuracy: {accuracy}')

# 获取验证集的预测结果
y_val_pred_prob = model.predict(X_val_scaled)
y_val_pred_encoded = np.argmax(y_val_pred_prob, axis=1)
y_val_pred = label_encoder.inverse_transform(y_val_pred_encoded)

# 计算验证集的评估指标
f1_val = f1_score(y_val, y_val_pred, average='weighted')
recall_val = recall_score(y_val, y_val_pred, average='weighted')
precision_val = precision_score(y_val, y_val_pred, average='weighted')
accuracy_val = accuracy_score(y_val, y_val_pred)

print("验证集评估指标：")
print("F1分数:", f1_val)
print("召回率:", recall_val)
print("精确率:", precision_val)
print("准确度:", accuracy_val)

# 预测测试集
X_test_scaled = scaler.transform(test[X.columns])
y_pred_prob = model.predict(X_test_scaled)
y_pred_encoded = np.argmax(y_pred_prob, axis=1)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# 将预测结果转换为 'yes' 或 'no' 字符串形式
y_pred = ['yes' if x == 1 else 'no' for x in y_pred]

# 创建提交的 DataFrame
submission = pd.DataFrame({
    'id': test['id'],  # 在这里使用测试数据集的 'id' 列的副本
    'subscribe': y_pred
})

# 保存提交文件
submission.to_csv('./submission-deeplearning.csv', index=False)