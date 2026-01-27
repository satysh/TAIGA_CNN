print("corsika_data module imported\n")

import struct              # чтение бинарных данных по байтам
import numpy as np         # массивы и численные операции
import torch               # PyTorch Dataset / DataLoader


class CustomDataset(torch.utils.data.Dataset):
    """
    Кастомный Dataset для PyTorch.
    Работает через индексы, чтобы не копировать данные.
    """
    def __init__(self, data, labels, indices):
        self.data = data          # полный массив изображений
        self.labels = labels      # полный массив меток
        self.indices = indices    # индексы (train или test)

    def __len__(self):
        # количество элементов в датасете
        return len(self.indices)

    def __getitem__(self, idx):
        # получаем реальный индекс в общем массиве
        real_idx = self.indices[idx]
        # возвращаем изображение и метку
        return self.data[real_idx], self.labels[real_idx]


class CorsikaData:
    """
    Класс для чтения бинарных файлов CORSIKA
    и формирования датасета для CNN.
    """

    header_size = 180            # размер заголовка события (байты)
    pixel_amplitude_size = 28    # размер одного пикселя (байты)

    def __init__(self):
        # data: (N, 1, 27, 27) — формат для Conv2D
        self.data = np.empty((0, 1, 27, 27), dtype=np.float32)
        # labels: 0 (proton) или 1 (gamma)
        self.labels = np.empty((0), dtype=np.int_)

        print("after init data shape", self.data.shape)
        print("after init labels shape", self.labels.shape)

    def load(self, files):
        """
        Загрузка списка бинарных файлов.
        """
        for file_name in files:
            self._load_single_file_(file_name)

        print("loading files done\n\n")

    def _load_single_file_(self, file_name):
        """
        Чтение одного бинарного файла.
        """
        print("loading file", file_name)
        event_counter = 0

        events = []    # временное хранилище изображений
        labels = []    # временное хранилище меток

        with open(file_name, "rb") as file_in_bytes:
            # читаем первый заголовок
            header_chunk = file_in_bytes.read(CorsikaData.header_size)

            # цикл по событиям
            while header_chunk:
                # 4 int32 — служебная информация (не используется)
                N_run, N_scattering, N_telescope, N_photoelectrons = \
                    struct.unpack('<4i', header_chunk[0:16])

                # 20 double — геометрия, энергия, тип частицы и т.д.
                (energy, theta, phi,
                 x_core, y_core, z_core,
                 h_1st_interaction, particle_type,
                 xmax, hmax,
                 x_telescope, y_telescope, z_telescope,
                 x_offset, y_offset,
                 theta_telescope, phi_telescope,
                 delta_alpha, alpha_pmt, T_average) = \
                    struct.unpack('<20d', header_chunk[16:176])

                # количество пикселей в событии
                N_pixels, = struct.unpack('<i', header_chunk[176:180])

                # изображение камеры 27x27
                tmp_event = np.zeros((27, 27))
                event_size = 0   # суммарная амплитуда события

                # чтение пикселей
                for i in range(N_pixels):
                    pixel_chunk = file_in_bytes.read(CorsikaData.pixel_amplitude_size)

                    # амплитуда и координаты пикселя
                    amplitude, row_number, column_number = \
                        struct.unpack('<3i', pixel_chunk[0:12])

                    # время (читается, но не используется)
                    average_time, std_time = \
                        struct.unpack('<2d', pixel_chunk[12:28])

                    # кладём амплитуду в центрированную матрицу
                    tmp_event[row_number + 13, column_number + 13] = amplitude
                    event_size += amplitude

                # отбор по суммарному сигналу
                if event_size > 120:
                    events.append(tmp_event)

                    # формирование метки класса
                    if np.isclose(particle_type, 1.):
                        labels.append(1)   # gamma
                    elif np.isclose(particle_type, 14.):
                        labels.append(0)   # proton
                    else:
                        raise ValueError("particle_type is not gamma or proton")

                event_counter += 1

                # читаем следующий заголовок
                header_chunk = file_in_bytes.read(CorsikaData.header_size)

        # проверка согласованности
        assert len(events) == len(labels)

        # добавление новых данных к уже загруженным
        print("\nbefore concatenate data shape", self.data.shape)
        print("before concatenate labels shape", self.labels.shape)

        self.data = np.concatenate([
            self.data,
            np.expand_dims(np.array(events, dtype=np.float32), axis=1)
        ])

        self.labels = np.concatenate([
            self.labels,
            np.array(labels, dtype=np.int_)
        ])

        print("after concatenate data shape", self.data.shape)
        print("after concatenate labels shape", self.labels.shape, '\n')

        # дополнительный фильтр "хороших" изображений
        is_good = []
        for i in range(len(self.data)):
            is_good.append(self._is_image_good_(self.data[i][0]))

        self.data = self.data[is_good]
        self.labels = self.labels[is_good]

        print("after goodness cut data shape", self.data.shape)
        print("after goodness cut labels shape", self.labels.shape)
        print("data.dtype:", self.data.dtype)
        print("labels.dtype:", self.labels.dtype, '\n\n')

    def _is_image_good_(self, random_image):
        """
        Проверка, что в изображении есть кластер
        из соседних ярких пикселей.
        """
        mask = random_image > 7          # порог по амплитуде
        center = mask[1:-1, 1:-1]        # исключаем края

        # проверка 8 соседей
        if np.any(center & mask[0:-2, 0:-2]): return True
        if np.any(center & mask[0:-2, 1:-1]): return True
        if np.any(center & mask[0:-2, 2:]):   return True
        if np.any(center & mask[1:-1, 0:-2]): return True
        if np.any(center & mask[1:-1, 2:]):   return True
        if np.any(center & mask[2:, 0:-2]):   return True
        if np.any(center & mask[2:, 1:-1]):   return True
        if np.any(center & mask[2:, 2:]):     return True

        return False

    def create_data_loader(
        self,
        batch_size: int = 128,
        train_portion: float = 0.8,
        shuffle_seed: int = 42
    ):
        """
        Создание train/test DataLoader'ов.
        """
        if len(self.data) != len(self.labels):
            raise ValueError(
                f"data and labels have different lengths: "
                f"{len(self.data)} and {len(self.labels)}"
            )

        # фиксированное перемешивание для воспроизводимости
        rng = np.random.default_rng(shuffle_seed)
        permut = rng.permutation(len(self.labels))

        train_portion_index = int(train_portion * len(self.labels))

        # индексы train/test
        self.train_indices = permut[:train_portion_index]
        self.test_indices = permut[train_portion_index:]

        # Dataset'ы через индексы
        train_dataset = CustomDataset(self.data, self.labels, self.train_indices)
        test_dataset = CustomDataset(self.data, self.labels, self.test_indices)

        # DataLoader'ы
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        print("create_data_loader done")
        return train_loader, test_loader
