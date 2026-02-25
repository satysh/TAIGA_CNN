import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

D_PMT = 15.0
SIN60 = np.sqrt(3) / 2


class EventVisualizer:

    def draw_event(self, dataset, event_idx, log_scale=False):
        image = dataset.data[event_idx, 0]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # ===== LEFT =====
        im = axes[0].imshow(
            np.log10(image + 1e-3) if log_scale else image,
            origin="lower",
            cmap="jet",
            extent=[-13, 13, -13, 13]
        )
        axes[0].set_aspect("equal")
        axes[0].set_title("CNN grid")
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        # ===== RIGHT =====
        patches = []
        colors = []
        xs_all = []
        ys_all = []

        for i in range(27):
            for j in range(27):

                Nr = i - 13
                Nc = j - 13

                # ПРАВИЛЬНАЯ гекс-сетка
                x = D_PMT * (Nc + 0.5 * (Nr & 1))
                y = D_PMT * Nr * SIN60

                xs_all.append(x)
                ys_all.append(y)

                hexagon = RegularPolygon(
                    (x, y),
                    numVertices=6,
                    radius=D_PMT / np.sqrt(3),
                    orientation=0
                )

                patches.append(hexagon)

                amp = image[i, j]
                if amp > 0:
                    colors.append(np.log10(amp + 1e-3) if log_scale else amp)
                else:
                    colors.append(np.nan)

        collection = PatchCollection(
            patches,
            cmap="jet",
            edgecolor="black",
            linewidth=0.3
        )

        arr = np.array(colors)
        collection.set_array(arr)
        collection.set_clim(
            vmin=np.nanmin(arr[np.isfinite(arr)]),
            vmax=np.nanmax(arr[np.isfinite(arr)])
        )

        axes[1].add_collection(collection)

        # центрирование
        lim = max(abs(np.array(xs_all)).max(),
                  abs(np.array(ys_all)).max())

        axes[1].set_xlim(-lim, lim)
        axes[1].set_ylim(-lim, lim)
        axes[1].set_aspect("equal")

        axes[1].set_title("PMT camera (correct hex geometry)")

        plt.colorbar(collection, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.show()
