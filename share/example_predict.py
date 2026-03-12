import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# список признаков (порядок важен)
features = [
    "width[0]",
    "length[0]",
    "alpha[0]",
    "miss[1]",
    "azwidth[1]",
    "dist[1]",
    "Xc[0]",
    "Yc[0]",
    "axis_scatter",
    "con2",
    "con_selected_island",
    "num_islands",
    "numb_pix",
]


# Можно загрузить модель напрямую из репозитория без клонирования.
import requests
import joblib
import io
url = "https://raw.githubusercontent.com/satysh/TAIGA_CNN/main/share/mlp_model_hillas.pkl"
response = requests.get(url)
model = joblib.load(io.BytesIO(response.content))

# Если модель скачана, то так 
# загрузка модели
#model = joblib.load("mpl_model_hillas.pkl")

# загрузка данных
data = pd.read_csv("data.csv")

# выбор признаков
X = data[features]

# если есть колонка с истинными метками
y_test = data["label"]

# предсказание
y_pred = model.predict(X)

print("Predictions:")
print(y_pred)

# вычисление метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1-score : {f1:.6f}")
