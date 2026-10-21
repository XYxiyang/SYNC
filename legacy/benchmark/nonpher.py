import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import joblib
import pickle
import warnings
import time
warnings.filterwarnings("ignore", category=FutureWarning)

## 生成示例数据：512维的分子指纹和二分类标签
#
np.random.seed(42)
#
#
## 随机生成分子指纹数据和二分类标签
#
#data = torch.load("data/processed/nonpher/170w.pt")
#test = torch.load("data/processed/nonpher/nonpher_test.pt")
#X_train = np.array([i['fp'] for i in data])
#y_train = np.array([i['y'] for i in data]).squeeze()
#X_test = np.array([i['fp'] for i in test])
#y_test = np.array([i['y'] for i in test]).squeeze()
#
#print(X_train.shape)
#print(y_train.shape)
#
## 创建随机森林分类器，树的数量为100
#
#rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=30)
#
## 训练模型
#rf_classifier.fit(X_train, y_train)
#
## 在测试集上进行预测
#y_pred = rf_classifier.predict(X_test)
#
## 输出分类结果和评估指标
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.4f}")
#print("\nClassification Report:")
#print(classification_report(y_test, y_pred))
#
#with open('random_forest_model.pkl', 'wb') as f:
#    pickle.dump(rf_classifier, f)
#print("Model saved to random_forest_model.pkl")



with open('random_forest_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print("Model loaded successfully!")
def test_sbdd():
    models = ["3dsbdd", "d3fg", "decompdiff", "diffbp", "diffsbdd", "flag", 
                  "graphbp", "ligan", "molcraft", "pocket2mol", "targetdiff", "voxbind", "enamine", "compas"]
    models = ["diffsbdd_caj"]
    models = ["enamine_1w"]
    models = ["targetdiff_dpo_1800"]
    models = ['compas_filter', 'coconut_5w']
    models = ['decomp_0510', 'decomp_dpo', 'decomp_origin']
    models = ['flowr', 'moldiff']
    models = ['neg_samples']

    models = ['8dyg_origin', '8dyg_dpo copy']

    for model in models:
        data = torch.load(f"data/sbdd/fps/{model}.pt")
        X_test = np.array([i['fp'] for i in data])
        y_test = np.array([i['y'] for i in data]).squeeze()
        y_pred = loaded_model.predict(X_test)
        print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model: {model}, Accuracy: {accuracy:.4f}")

#for idx in range(1, 4):
#    data = torch.load(f"data/processed/nonpher/gasa_test{idx}.pt")
#    print(data[0]['fp'].shape)  
#    X_test = np.array([i['fp'] for i in data])
#    y_test = np.array([i['y'] for i in data]).squeeze()
#    y_pred = loaded_model.predict(X_test)
#    accuracy = accuracy_score(y_test, y_pred)
#    print(f"Model: gasa_test{idx}, Accuracy: {accuracy:.4f}")
start_time = time.time()
test_sbdd()
end_time = time.time()
print(f"Validation time: {end_time - start_time}")