# Знакомство со Сверточной нейронной сетью из torch

base_v - ветка с кодом Елдоса <br>
main - основная ветка с моими попытками 

# CNN training notebook (`test.ipynb`)

Этот ноутбук используется для обучения сверточной нейросети
для бинарной классификации событий (gamma / proton).

Логика обучения во всех случаях одинаковая.
Меняется только способ подготовки входных данных.

---

## Общая схема работы

Данные проходят следующий путь:

Данные → Dataset → DataLoader → SimpleConvNet → обучение → тест → метрики

Если данные на входе модели имеют правильный формат,
ноутбук работает без изменений.

---

## Формат данных на входе модели

На вход нейросети всегда должно подаваться:

- `data` — тензор формы `(batch, 1, 27, 27)`
- `labels` — метки классов `0` или `1`
- тип данных — `float32`

Все источники данных должны быть приведены к этому виду.

---

## Использование бинарных файлов CORSIKA

Это стандартный вариант работы.

Используется класс `CorsikaData`, который:
- читает бинарные файлы
- преобразует каждое событие в изображение `27×27`
- делит данные на обучающую и тестовую выборки

Пример:

```python
dataset = CorsikaData()
dataset.load(files)

train_loader, test_loader = dataset.create_data_loader(
    batch_size=256,
    train_portion=0.95
)

---

Main goal of our telescopes is to detect gamma-quants from various astrophysical sources. The difficulty is that ratio between gamma flux and hadron flux is about 10^-5. Therefore, we need to filter out as much of  the hadronic background as possible and it seems that convolution and/or graph neural networks could be useful in this task.

About the dataset:
I'm attaching a dataset with simulated gamma and hadronic events. The structure is as follows: 
- gamma/ - contains simulated gamma-quanta events;
- proton/ - contains hadron (proton) background events.

Both directories have same internal structure and contains event data in next representations:
1. Raw simulated data (all-pixel, cherenkov photons from air-shower only)
2. Raw data + background (all-pixel, simulated event is overlapped with sky noise)
3. Cleaned data (only bright pixels after amplitude-based threshold cleaning)

Inner structure of the folders:
- bpe* - parent folder with given simulated dataset. There are several datasets with different energy ranges and arrival angles. As a first approximation you can just stack it, later it will be needed to renormalize it according to energy/amplitude.
- bpe*/trig0000/*_c.txt - all-pixel data, cherenkov photons only.
- bpe*/trig0000/b0/*_cb0.txt - all-pixel data, cherenkov photons + sky noise, close to real measurements
- bpe*/trig0000/b0/14-7fix/*_clean_*_cb0.txt - pixel data after cleaning (only high-amplitude pixels)
- bpe*/trig0000/b0/14-7fix/*_hillas_*_cb0.txt - high-level parameters of events (see description in the report attached in archive)

Pixel data files structure:
- Header is one line with meta parameters of event, it followed by N number of lines, each line corresponds to single pixel. After N lines there are next header and so on. 
- Header structure slightly differ for raw and cleaned data, please look to raw full-pixel data files (bpe*/trig0000/b0/*_cb0.txt ).

Header fields (as in bpe*/trig0000/b0/*_cb0.txt):
1. Event number in CORSIKA simulation
2. Number of axis scattering (e.g. number of copy of an event with varied axis position, used for increasing the dataset)
3. N, number of pixels in event 
4. Primary particle energy (TeV)
5. X-coordinate of shower axis related to telescope position, (meters)
6. Y-coordinate of shower axis related to telescope position, (meters)
7. Telescope zenith angle (radians)
8. Telescope azimuth angle (radians)
9. Air-shower axis zenith angle (radians)
10. Air-shower axis azimuth angle (radians)
11. Atmospheric depth of shower maximum (g/cm^2)

Pixel line structure:
1. Cluster number (pixels are organized into clusters of 28)
2. Pixel number within the cluster
3. X-coordinate of pixel in camera
4. Y-coordinate of pixel in camera
5. Amplitude. Note: Negative amplitudes are possible due to mean value (pedestal) subtraction.
