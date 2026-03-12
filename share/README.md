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

## 3. Результаты на данных, котрые модель не видела, но которые взяты из той же выборки, что и данные на которых модель училась. 

Протоны: bpe5432_32m_da5.0_md52021/trig0000/b0/

Гамма: bpe607_31_da0.0_md5_old_cone

Accuracy : 0.993444

Precision: 0.995188

Recall   : 0.990421

F1-score : 0.992799

Test ROC AUC: 0.9989281886387996

---



