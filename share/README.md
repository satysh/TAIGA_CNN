# MLP Hillas Classifier

Модель классификации, обученная на признаках Hillas.
Используется для предсказания класса события на основе набора физических признаков.

# Описание модели

Модель принимает DataFrame: **13 признаков**:

```
width[0]
length[0]
alpha[0]
miss[1]
azwidth[1]
dist[1]
Xc[0]
Yc[0]
axis_scatter
con2
con_selected_island
num_islands
numb_pix
```

⚠️ **Порядок признаков важен.**
Данные должны подаваться в модель **строго в этом порядке**.

---

# Требования

Python ≥ 3.9

Основные библиотеки:

```
numpy
pandas
scikit-learn
joblib
```

Установка:

```bash
pip install numpy pandas scikit-learn joblib
```

---

# Структура репозитория

```
.
├── mpl_model_hillas.pkl
├── README.md
└── example_predict.py
```

---

# Загрузка модели

```python
import joblib

model = joblib.load("mlp_model_hillas.pkl")
```

---

# Использование модели

## 1. Подготовка признаков

```python
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
```

---

## 2. Предсказание

```python
import pandas as pd
import joblib

model = joblib.load("mlp_model_hillas.pkl")

df = pd.read_csv("data.csv")

X = df[features]

predictions = model.predict(X)

print(predictions)
```

---



