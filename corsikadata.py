import struct
from pathlib import Path

import numpy as np
import torch


class CustomDataset(torch.utils.data.Dataset):
    """Кастомный Dataset для PyTorch (по индексам, без копирования массива)."""

    def __init__(self, data, labels, indices):
        self.data = data
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = torch.as_tensor(self.data[real_idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[real_idx], dtype=torch.float32)
        return image, label


class CorsikaData:
    """Чтение бинарных файлов CORSIKA и подготовка датасета для CNN."""

    HEADER_SIZE = 180
    PIXEL_AMPLITUDE_SIZE = 28
    IMAGE_SIZE = 27
    IMAGE_CENTER_SHIFT = 13
    EVENT_SIZE_THRESHOLD = 120
    PIXEL_BRIGHT_THRESHOLD = 7

    def __init__(self, debug_first_events=0):
        self.data = np.empty((0, 1, self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.float32)
        self.train_indices = np.array([], dtype=np.int64)
        self.test_indices = np.array([], dtype=np.int64)
        self.debug_first_events = max(int(debug_first_events), 0)

    def load(self, files):
        for file_name in files:
            self._load_single_file(file_name)

    def _load_single_file(self, file_name):
        file_name = Path(file_name)
        events = []
        labels = []

        event_counter = 0

        with file_name.open("rb") as file_in_bytes:
            header_chunk = file_in_bytes.read(self.HEADER_SIZE)

            while header_chunk:
                # 4 int32 — служебная информация (сейчас не используется)
                _n_run, _n_scattering, _n_telescope, _n_photoelectrons = struct.unpack(
                    "<4i", header_chunk[0:16]
                )
                header_values = struct.unpack("<20d", header_chunk[16:176])
                particle_type = header_values[7]
                (n_pixels,) = struct.unpack("<i", header_chunk[176:180])

                event = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float32)
                event_size = 0

                for pixel_idx in range(n_pixels):
                    pixel_chunk = file_in_bytes.read(self.PIXEL_AMPLITUDE_SIZE)
                    amplitude, row_number, column_number = struct.unpack("<3i", pixel_chunk[0:12])

                    if event_counter < self.debug_first_events:
                        print(
                            f"[event {event_counter:3d}] "
                            f"pix {pixel_idx:3d} | amp={amplitude:4d} "
                            f"| row={row_number:3d} col={column_number:3d}"
                        )

                    row = row_number + self.IMAGE_CENTER_SHIFT
                    col = column_number + self.IMAGE_CENTER_SHIFT
                    if 0 <= row < self.IMAGE_SIZE and 0 <= col < self.IMAGE_SIZE:
                        event[row, col] = amplitude
                        event_size += amplitude

                if event_size > self.EVENT_SIZE_THRESHOLD:
                    events.append(event)

                    if np.isclose(particle_type, 1.0):
                        labels.append(1.0)
                    elif np.isclose(particle_type, 14.0):
                        labels.append(0.0)
                    else:
                        raise ValueError(f"Unsupported particle_type: {particle_type}")

                event_counter += 1

                header_chunk = file_in_bytes.read(self.HEADER_SIZE)

        if len(events) != len(labels):
            raise RuntimeError("Loaded events and labels count mismatch")

        if events:
            file_data = np.expand_dims(np.array(events, dtype=np.float32), axis=1)
            file_labels = np.array(labels, dtype=np.float32)

            self.data = np.concatenate([self.data, file_data], axis=0)
            self.labels = np.concatenate([self.labels, file_labels], axis=0)

        if len(self.data) > 0:
            is_good = np.array([self._is_image_good(img[0]) for img in self.data], dtype=bool)
            self.data = self.data[is_good]
            self.labels = self.labels[is_good]

    def _is_image_good(self, image):
        mask = image > self.PIXEL_BRIGHT_THRESHOLD
        center = mask[1:-1, 1:-1]

        return bool(
            np.any(center & mask[0:-2, 0:-2])
            or np.any(center & mask[0:-2, 1:-1])
            or np.any(center & mask[0:-2, 2:])
            or np.any(center & mask[1:-1, 0:-2])
            or np.any(center & mask[1:-1, 2:])
            or np.any(center & mask[2:, 0:-2])
            or np.any(center & mask[2:, 1:-1])
            or np.any(center & mask[2:, 2:])
        )

    def create_data_loader(self, batch_size=128, train_portion=0.8, shuffle_seed=42):
        if not 0 < train_portion < 1:
            raise ValueError("train_portion must be in range (0, 1)")

        if len(self.data) != len(self.labels):
            raise ValueError(
                f"data and labels have different lengths: {len(self.data)} and {len(self.labels)}"
            )

        rng = np.random.default_rng(shuffle_seed)
        permutation = rng.permutation(len(self.labels))

        train_end = int(train_portion * len(self.labels))
        self.train_indices = permutation[:train_end]
        self.test_indices = permutation[train_end:]

        train_dataset = CustomDataset(self.data, self.labels, self.train_indices)
        test_dataset = CustomDataset(self.data, self.labels, self.test_indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )

        return train_loader, test_loader
