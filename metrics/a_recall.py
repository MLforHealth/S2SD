import numpy as np

class Metric():
    def __init__(self, k, **kwargs):
        self.k = k
        self.requires = ['kmeans', 'nearest']

    def compute(self, target_labels, k_closest_classes):
        recall_all_k = []
        for k in k_vals:
            recall_at_k = np.sum([1 for target, recalled_predictions in zip(target_labels, k_closest_classes) if target in recalled_predictions[:k]])/len(target_labels)
            recall_all_k.append(recall_at_k)
        return recall_all_k
