import matplotlib.pyplot as plt
import numpy as np


class Metrics:
    def draw_precision_from_suppression(
        self,
        y_probabilities,
        test_labels,
        gate_start_value,
        gate_n,
    ):
        y_probabilities = np.asarray(y_probabilities).reshape(-1)
        test_labels = np.asarray(test_labels).reshape(-1)

        if y_probabilities.shape[0] != test_labels.shape[0]:
            raise ValueError("y_probabilities and test_labels must have the same length")

        gate_values_array = np.linspace(gate_start_value, 1.0, gate_n)

        precision_array = []
        suppression_array = []

        for gate in gate_values_array:
            y_after_gate = (y_probabilities >= gate).astype(np.int32)

            true_positives = np.sum((y_after_gate == 1) & (test_labels == 1))
            false_negatives = np.sum((y_after_gate == 0) & (test_labels == 1))
            false_positives = np.sum((y_after_gate == 1) & (test_labels == 0))
            true_negatives = np.sum((y_after_gate == 0) & (test_labels == 0))

            precision_denom = true_positives + false_negatives
            precision = true_positives / precision_denom if precision_denom > 0 else 0.0

            # Inverse of False Omission Rate.
            suppression = (
                (false_positives + true_negatives) / false_positives
                if false_positives > 0
                else np.inf
            )

            precision_array.append(precision)
            suppression_array.append(suppression)

        plt.plot(precision_array, suppression_array, linewidth=3)
        plt.ylabel("Suppression", fontsize=18)
        plt.grid(True, which="both")
        plt.xlabel("Precision", fontsize=18)
        plt.title("ROC", fontsize=20)
        plt.show()

        return precision_array, suppression_array
