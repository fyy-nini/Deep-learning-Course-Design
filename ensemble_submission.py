import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from autogluon.tabular import TabularDataset, TabularPredictor

# 加载数据
train = pd.read_csv('downloads/136244/train.csv')
test = pd.read_csv('downloads/136244/test.csv')

# 重新定义特征，基于两个数据集共有的、且不包含 'subscribe' 的列
common_columns = [col for col in train.columns if col in test.columns and col!= 'subscribe']
features = common_columns

# 处理字符串类型的特征（训练集）
X = pd.get_dummies(train[features])

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, train['subscribe'], test_size=0.2, random_state=42)

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

# 构建神经网络模型（这里可以考虑添加超参数调整相关代码，比如尝试不同的层数、神经元数量、学习率等）
model = Sequential()
model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train_scaled, y_train_onehot, epochs=100, batch_size=32, validation_data=(X_val_scaled, y_val_onehot), verbose=1)

# 评估深度学习模型在验证集上的表现
loss, accuracy = model.evaluate(X_val_scaled, y_val_onehot)
print(f'深度学习模型 - 验证集 Accuracy: {accuracy}')
y_val_pred_prob = model.predict(X_val_scaled)
y_val_pred_encoded = np.argmax(y_val_pred_prob, axis=1)
y_val_pred_deeplearning = label_encoder.inverse_transform(y_val_pred_encoded)

# 计算深度学习模型验证集的评估指标
f1_val_deeplearning = f1_score(y_val, y_val_pred_deeplearning, average='weighted')
recall_val_deeplearning = recall_score(y_val, y_val_pred_deeplearning, average='weighted')
precision_val_deeplearning = precision_score(y_val, y_val_pred_deeplearning, average='weighted')
accuracy_val_deeplearning = accuracy_score(y_val, y_val_pred_deeplearning)

print("深度学习模型 - 验证集评估指标：")
print("F1分数:", f1_val_deeplearning)
print("召回率:", recall_val_deeplearning)
print("精确率:", precision_val_deeplearning)
print("准确度:", accuracy_val_deeplearning)

# 使用AutoGluon训练并获取预测结果（这里获取在验证集和测试集上的结果）
train_data = TabularDataset(pd.concat([X_train, y_train], axis=1))
id_col = 'id'
label_col = 'subscribe'
predictor = TabularPredictor(label=label_col).fit(train_data.drop(columns=[id_col]))

# 在验证集上获取AutoGluon预测结果
X_val_for_autogluon = X_val.copy()
autogluon_preds_val = predictor.predict(X_val_for_autogluon)

# 处理测试集特征（确保和训练集特征处理逻辑一致）
# 先处理测试集可能缺失的列情况
missing_cols_in_test = set(features) - set(test.columns)
for col in missing_cols_in_test:
    test[col] = np.nan

X_test = pd.get_dummies(test[features])

# 在测试集上获取AutoGluon预测结果
X_test_for_autogluon = X_test.copy()
autogluon_preds_test = predictor.predict(X_test_for_autogluon)

# 融合预测结果前先处理类型转换（将字符串类型的预测结果转换为数值类型）
# 对深度学习模型预测结果进行类型转换
y_val_pred_deeplearning_numeric = []
for val in y_val_pred_deeplearning:
    if val == 'yes':
        y_val_pred_deeplearning_numeric.append(1)
    elif val == 'no':
        y_val_pred_deeplearning_numeric.append(0)
    else:
        raise ValueError(f"Unexpected value: {val}")
y_val_pred_deeplearning_numeric = np.array(y_val_pred_deeplearning_numeric)

# 对AutoGluon预测结果进行类型转换（确保严格转换为数值类型）
autogluon_preds_val_numeric = []
for val in autogluon_preds_val:
    if val == 'yes':
        autogluon_preds_val_numeric.append(1)
    elif val == 'no':
        autogluon_preds_val_numeric.append(0)
    else:
        raise ValueError(f"Unexpected value: {val}")
autogluon_preds_val_numeric = np.array(autogluon_preds_val_numeric)

# 对测试集的AutoGluon预测结果也进行类型转换
autogluon_preds_test_numeric = []
for val in autogluon_preds_test:
    if val == 'yes':
        autogluon_preds_test_numeric.append(1)
    elif val == 'no':
        autogluon_preds_test_numeric.append(0)
    else:
        raise ValueError(f"Unexpected value: {val}")
autogluon_preds_test_numeric = np.array(autogluon_preds_test_numeric)

# 融合预测结果（简单示例，这里可以采用加权平均等更合理的融合策略，比如根据各自模型在验证集上的表现确定权重，也可以尝试其他复杂融合策略比如堆叠融合等）
# 假设简单等权重融合（0.5, 0.5），实际需优化权重选择
combined_preds_val = (y_val_pred_deeplearning_numeric + autogluon_preds_val_numeric) / 2
combined_preds_val = combined_preds_val.round().astype(int)

# 对测试集预测结果进行融合操作
combined_preds_test = (y_pred_deeplearning_numeric + autogluon_preds_test_numeric) / 2
combined_preds_test = combined_preds_test.round().astype(int)

# 将融合后的预测结果转换为 'yes' 或 'no' 字符串形式（假设预测结果是0或1相关格式，根据实际调整）
combined_preds_val = ['yes' if x == 1 else 'no' for x in combined_preds_val]
combined_preds_test = ['yes' if x == 1 else 'no' for x in combined_preds_test]

# 计算融合后模型在验证集上的评估指标
f1_val_combined = f1_score(y_val, combined_preds_val, average='weighted')
recall_val_combined = recall_score(y_val, combined_preds_val, average='weighted')
precision_val_combined = precision_score(y_val, combined_preds_val, average='weighted')
accuracy_val_combined = accuracy_score(y_val, combined_preds_val)

print("融合模型 - 验证集评估指标：")
print("F1分数:", f1_val_combined)
print("召回率:", recall_val_combined)
print("精确率:", precision_val_combined)
print("准确度:", accuracy_val_combined)

# 深度学习模型预测测试集并生成结果文件（这里可以考虑添加更多对预测结果的分析或者保存更多相关信息等）
y_pred_deeplearning_prob = model.predict(scaler.transform(X_test))
y_pred_deeplearning_encoded = np.argmax(y_pred_deeplearning_prob, axis=1)
y_pred_deeplearning_result = label_encoder.inverse_transform(y_pred_deeplearning_encoded)
y_pred_deeplearning_result = ['yes' if x == 1 else 'no' for x in y_pred_deeplearning_result]
submission_deeplearning = pd.DataFrame({
    'id': test['id'],
    'subscribe': y_pred_deeplearning_result
})
submission_deeplearning.to_csv('./submission_deeplearning.csv', index=False)

# AutoGluon模型预测测试集并生成结果文件
autogluon_preds_test = predictor.predict(X_test)
autogluon_preds_test = ['yes' if x == 1 else 'no' for x in autogluon_preds_test]
submission_autogluon = pd.DataFrame({
    'id': test['id'],
    'subscribe': autogluon_preds_test
})
submission_autogluon.to_csv('./submission_autogluon.csv', index=False)

# 创建提交的融合模型结果DataFrame（使用融合后的测试集预测结果）
submission = pd.DataFrame({
    'id': test['id'],
    'subscribe': combined_preds_test
})

# 保存提交文件
submission.to_csv('./ensemble_submission.csv', index=False)