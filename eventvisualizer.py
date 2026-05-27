import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection


class EventVisualizer:
    """
    Универсальный визуализатор:
      - поддерживает новый CleanEventDataset (через build_cnn_image/get_event_pixels)
      - сохраняет совместимость со старым интерфейсом (dataset.data)
    """

    def draw_event(self, dataset, event_idx, log_scale=False, center_mode="mean"):
        # --- Новый формат датасета ---
        if hasattr(dataset, "build_cnn_image") and hasattr(dataset, "get_event_pixels"):
            image = dataset.build_cnn_image(event_idx, center_mode=center_mode)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # LEFT: CNN grid
            im = axes[0].imshow(
                np.log10(image + 1e-3) if log_scale else image,
                origin="lower",
                cmap="jet",
                extent=[-13, 13, -13, 13],
            )
            axes[0].set_aspect("equal")
            axes[0].set_title(f"CNN grid (27x27 pad), centered={center_mode}")
            plt.colorbar(im, ax=axes[0], fraction=0.046)

            # RIGHT: реальная геометрия камеры
            x, y, a = dataset.get_event_pixels(event_idx)
            cam_colors = np.zeros(len(dataset.camera_xy), dtype=np.float64)
            cam_xy_arr = np.array(dataset.camera_xy, dtype=np.float64)

            for xi, yi, ai in zip(x, y, a):
                key = dataset._xy_key(xi, yi)
                idx = dataset.camera_key_to_idx.get(key, None)
                if idx is None:
                    # fallback: nearest pixel in camera geometry (защита от ошибок округления)
                    d2 = (cam_xy_arr[:, 0] - float(xi)) ** 2 + (cam_xy_arr[:, 1] - float(yi)) ** 2
                    idx = int(np.argmin(d2))
                if ai > 0:
                    cam_colors[idx] = np.log10(ai + 1e-3) if log_scale else ai

            patches = []
            for (cx, cy) in dataset.camera_xy:
                patches.append(
                    RegularPolygon(
                        (cx, cy),
                        numVertices=6,
                        radius=dataset.hex_radius,
                        orientation=0,
                    )
                )

            cmap = plt.cm.get_cmap("jet").copy()
            cmap.set_bad(color="lightgray")

            collection = PatchCollection(
                patches,
                cmap="jet",
                edgecolor="black",
                linewidth=0.2,
            )
            collection.set_array(cam_colors)

            finite = np.isfinite(cam_colors)
            if np.any(finite):
                collection.set_clim(
                    np.nanmin(cam_colors[finite]),
                    np.nanmax(cam_colors[finite]),
                )
            axes[1].add_collection(collection)

            cam_xy = np.array(dataset.camera_xy, dtype=np.float64)
            xmin, ymin = cam_xy.min(axis=0)
            xmax, ymax = cam_xy.max(axis=0)
            pad = dataset.pitch * 2.0

            axes[1].set_xlim(xmin - pad, xmax + pad)
            axes[1].set_ylim(ymin - pad, ymax + pad)
            axes[1].set_aspect("equal")
            axes[1].set_title("PMT camera (full mask, fixed scale)")
            plt.colorbar(collection, ax=axes[1], fraction=0.046)

            plt.tight_layout()
            plt.show()
            return

        # --- Старый формат датасета (обратная совместимость) ---
        if hasattr(dataset, "data"):
            image = dataset.data[event_idx, 0]

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # LEFT: исходная CNN-сетка 27x27
            im = axes[0].imshow(
                np.log10(image + 1e-3) if log_scale else image,
                origin="lower",
                cmap="jet",
                extent=[-13, 13, -13, 13],
            )
            axes[0].set_aspect("equal")
            axes[0].set_title("CNN grid (27x27)")
            plt.colorbar(im, ax=axes[0], fraction=0.046)

            # RIGHT: оценка hex-камеры напрямую из row/col индексов
            # row/col в CorsikaData лежат в диапазоне [-13..13] и центрированы относительно 0.
            rows, cols = np.nonzero(image > 0)
            amps = image[rows, cols].astype(np.float64)
            rows = rows.astype(np.int64) - 13
            cols = cols.astype(np.int64) - 13

            if rows.size > 0:
                pitch = 1.0
                dy = np.sqrt(3.0) / 2.0
                hex_radius = 1.0 / np.sqrt(3.0)

                xs = cols.astype(np.float64) * pitch + 0.5 * pitch * (rows & 1)
                ys = rows.astype(np.float64) * dy

                patches = [
                    RegularPolygon((x, y), numVertices=6, radius=hex_radius, orientation=0)
                    for x, y in zip(xs, ys)
                ]
                colors = np.log10(amps + 1e-3) if log_scale else amps

                coll = PatchCollection(
                    patches,
                    cmap="jet",
                    edgecolor="black",
                    linewidth=0.25,
                )
                coll.set_array(colors)
                coll.set_clim(np.nanmin(colors), np.nanmax(colors))
                axes[1].add_collection(coll)

                pad = 1.5
                axes[1].set_xlim(float(np.min(xs) - pad), float(np.max(xs) + pad))
                axes[1].set_ylim(float(np.min(ys) - pad), float(np.max(ys) + pad))
                axes[1].set_aspect("equal")
                axes[1].set_title("Estimated PMT hex camera")
                plt.colorbar(coll, ax=axes[1], fraction=0.046)
            else:
                axes[1].set_title("Estimated PMT hex camera (no non-zero pixels)")
                axes[1].set_aspect("equal")

            plt.tight_layout()
            plt.show()
            return

        raise AttributeError(
            "Unsupported dataset format: expected either "
            "(build_cnn_image + get_event_pixels) or data attribute"
        )