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
            cam_colors = np.full(len(dataset.camera_xy), np.nan, dtype=np.float64)

            for xi, yi, ai in zip(x, y, a):
                key = dataset._xy_key(xi, yi)
                idx = dataset.camera_key_to_idx.get(key, None)
                if idx is None:
                    continue
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
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            im = ax.imshow(
                np.log10(image + 1e-3) if log_scale else image,
                origin="lower",
                cmap="jet",
                extent=[-13, 13, -13, 13],
            )
            ax.set_aspect("equal")
            ax.set_title("CNN grid")
            plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            plt.show()
            return

        raise AttributeError(
            "Unsupported dataset format: expected either "
            "(build_cnn_image + get_event_pixels) or data attribute"
        )
