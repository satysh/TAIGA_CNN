import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection


class CleanEventDataset:
    """
    Читает *_clean_*.txt:
      header: 11 чисел, header[3] = N
      далее N строк: cluster, pix_id, x, y, amp

    Главное:
      - PMT-геометрию НЕ "угадываем" по шагам.
        Берём реальную камеру как множество всех уникальных (x,y) из файла.
      - CNN-репрезентация: 27x27 паддинг + центрирование события.
    """

    def __init__(self, path: str, camera_events_for_geometry: int = 500):
        self.path = path
        self.events = []

        self._key_decimals = 5   # <-- ПЕРЕМЕСТИТЬ СЮДА

        self._parse(path)

        # Геометрия камеры
        self.camera_xy = self._build_camera_geometry(n_events=camera_events_for_geometry)

        self.camera_key_to_idx = {
            self._xy_key(x, y): i for i, (x, y) in enumerate(self.camera_xy)
        }

        self.pitch = self._estimate_pitch(self.camera_xy)
        self.hex_radius = self.pitch / np.sqrt(3)
        self.dy = self._estimate_dy(self.camera_xy)


    def __len__(self):
        return len(self.events)

    # -------------------- PARSER --------------------

    def _parse(self, path):
        with open(path, "r") as f:
            lines = f.readlines()

        i = 0
        L = len(lines)

        while i < L:
            line = lines[i].strip()
            i += 1
            if not line:
                continue

            parts = line.split()
            if len(parts) != 11:
                continue

            header = np.array([float(x) for x in parts], dtype=np.float64)
            n_pix = int(round(header[3]))

            pix = np.zeros((n_pix, 5), dtype=np.float32)
            for k in range(n_pix):
                vals = lines[i].strip().split()
                i += 1
                if len(vals) != 5:
                    raise ValueError(f"Bad pixel line near line {i}: {lines[i-1]}")
                pix[k] = np.array([float(x) for x in vals], dtype=np.float32)

            self.events.append({"header": header, "pixels": pix})

    # -------------------- GEOMETRY HELPERS --------------------

    def _xy_key(self, x, y):
        return (round(float(x), self._key_decimals), round(float(y), self._key_decimals))

    def _build_camera_geometry(self, n_events: int = 500):
        """
        Камера = все уникальные (x,y), которые встречаются в первых n_events событий.
        Этого обычно хватает, чтобы собрать всю маску камеры.
        """
        xs = []
        ys = []
        take = min(len(self.events), n_events)
        for ev in self.events[:take]:
            xs.append(ev["pixels"][:, 2])
            ys.append(ev["pixels"][:, 3])

        xs = np.concatenate(xs) if xs else np.array([], dtype=np.float32)
        ys = np.concatenate(ys) if ys else np.array([], dtype=np.float32)

        xy = np.stack([xs, ys], axis=1)
        xy = np.round(xy, self._key_decimals)

        # уникальные точки камеры
        xy_unique = np.unique(xy, axis=0)
        # для красивой отрисовки можно отсортировать
        xy_unique = xy_unique[np.lexsort((xy_unique[:, 0], xy_unique[:, 1]))]

        return [(float(x), float(y)) for x, y in xy_unique]

    def _estimate_pitch(self, camera_xy):
        """
        Оценка расстояния между ближайшими центрами (pitch) по всем точкам камеры.
        Простая O(N^2) на умеренном числе пикселей. Для TAIGA обычно норм.
        """
        pts = np.array(camera_xy, dtype=np.float64)
        n = pts.shape[0]
        if n < 2:
            return 1.0

        # грубо: для каждого i ищем ближайшего соседа
        min_d = np.inf
        for i in range(n):
            dx = pts[:, 0] - pts[i, 0]
            dy = pts[:, 1] - pts[i, 1]
            d2 = dx * dx + dy * dy
            d2[i] = np.inf
            j = np.argmin(d2)
            d = np.sqrt(d2[j])
            if d > 1e-9 and d < min_d:
                min_d = d

        # fallback
        if not np.isfinite(min_d) or min_d <= 0:
            return 1.0
        return float(min_d)

    def _estimate_dy(self, camera_xy):
        """
        Оценка шага между "рядами" по y (минимальный ненулевой diff y).
        В твоих данных это около 2.598.
        """
        ys = np.unique(np.round(np.array([y for _, y in camera_xy], dtype=np.float64), self._key_decimals))
        diffs = np.diff(np.sort(ys))
        diffs = diffs[diffs > 1e-6]
        if diffs.size == 0:
            return 1.0
        return float(np.min(diffs))

    # -------------------- EVENT ACCESS --------------------

    def get_event_pixels(self, event_idx):
        """
        Возвращает пиксели события в "камера-координатах" (без центрирования!):
          x, y, amp
        """
        pix = self.events[event_idx]["pixels"]
        x = pix[:, 2].astype(np.float64)
        y = pix[:, 3].astype(np.float64)
        a = pix[:, 4].astype(np.float64)
        return x, y, a

    # -------------------- CNN REPRESENTATION --------------------

    def build_cnn_image(self, event_idx, center_mode="mean"):

        x, y, a = self.get_event_pixels(event_idx)

        # Получаем индексы камеры для пикселей события
        rows = []
        cols = []

        for xi, yi in zip(x, y):
            key = self._xy_key(xi, yi)
            idx = self.camera_key_to_idx.get(key, None)
            if idx is None:
                continue

            # камера_xy уже отсортирована по y, затем x
            # разложим её в условные row/col через dy
            cx, cy = self.camera_xy[idx]

            # row определяется по y
            row = int(round(cy / self.dy))
            # col определяется по x (с учётом hex-сдвига)
            col = int(round((cx - 0.5 * self.pitch * (row & 1)) / self.pitch))

            rows.append(row)
            cols.append(col)

        rows = np.array(rows)
        cols = np.array(cols)

        if len(rows) == 0:
            return np.zeros((27, 27), dtype=np.float32)

        # --- центрирование в индексном пространстве ---
        if center_mode == "mean":
            r0 = int(round(np.mean(rows)))
            c0 = int(round(np.mean(cols)))
        elif center_mode == "max":
            imax = np.argmax(a)
            r0 = rows[imax]
            c0 = cols[imax]
        else:
            r0 = 0
            c0 = 0

        img = np.zeros((27, 27), dtype=np.float32)

        for r, c, amp in zip(rows, cols, a):
            rr = r - r0 + 13
            cc = c - c0 + 13

            if 0 <= rr < 27 and 0 <= cc < 27:
                img[rr, cc] += float(amp)

        return img



class EventVisualizer:
    """
    Рисует 2 панели:
      - слева: CNN grid (27x27, со сдвигом)
      - справа: реальная камера (вся маска), без сдвига, фиксированный масштаб
    """

    def draw_event(self, dataset: CleanEventDataset, event_idx: int,
                   log_scale: bool = False,
                   center_mode: str = "mean"):

        # --- CNN image (centered)
        img = dataset.build_cnn_image(event_idx, center_mode=center_mode)
        img_show = np.log10(img + 1e-3) if log_scale else img

        # --- PMT camera values (NOT centered)
        x, y, a = dataset.get_event_pixels(event_idx)

        # подготовим цвета для всей камеры
        cam_colors = np.full(len(dataset.camera_xy), np.nan, dtype=np.float64)

        for xi, yi, ai in zip(x, y, a):
            key = dataset._xy_key(xi, yi)
            idx = dataset.camera_key_to_idx.get(key, None)
            if idx is None:
                continue
            if ai > 0:
                cam_colors[idx] = (np.log10(ai + 1e-3) if log_scale else ai)
            else:
                cam_colors[idx] = np.nan

        # --- plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # LEFT
        im = axes[0].imshow(
            img_show,
            origin="lower",
            cmap="jet",
            extent=[-13, 13, -13, 13]
        )
        axes[0].set_aspect("equal")
        axes[0].set_title(f"CNN grid (27x27 pad), centered={center_mode}")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        # RIGHT: full camera
        patches = []
        for (cx, cy) in dataset.camera_xy:
            hexagon = RegularPolygon(
                (cx, cy),
                numVertices=6,
                radius=dataset.hex_radius,
                orientation=0
            )
            patches.append(hexagon)

        coll = PatchCollection(
            patches,
            cmap="jet",
            edgecolor="black",
            linewidth=0.2
        )
        coll.set_array(cam_colors)

        finite = np.isfinite(cam_colors)
        if np.any(finite):
            coll.set_clim(np.nanmin(cam_colors[finite]), np.nanmax(cam_colors[finite]))

        axes[1].add_collection(coll)

        # фиксированный масштаб по камере (не по событию)
        cam_xy = np.array(dataset.camera_xy, dtype=np.float64)
        xmin, ymin = cam_xy.min(axis=0)
        xmax, ymax = cam_xy.max(axis=0)

        pad = dataset.pitch * 2.0
        axes[1].set_xlim(xmin - pad, xmax + pad)
        axes[1].set_ylim(ymin - pad, ymax + pad)
        axes[1].set_aspect("equal")
        axes[1].set_title("PMT camera (full mask, fixed scale)")

        plt.colorbar(coll, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.show()
