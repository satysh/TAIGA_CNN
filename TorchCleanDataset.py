import torch

class TorchCleanDataset(torch.utils.data.Dataset):
    def __init__(self, gamma_datasets, proton_datasets):
        """
        gamma_datasets  – список датасетов с гамма-событиями
        proton_datasets – список датасетов с протон-событиями
        """

        # если передали один объект — обернём в список
        if not isinstance(gamma_datasets, (list, tuple)):
            gamma_datasets = [gamma_datasets]

        if not isinstance(proton_datasets, (list, tuple)):
            proton_datasets = [proton_datasets]

        self.gamma_datasets = gamma_datasets
        self.proton_datasets = proton_datasets

        # считаем длины
        self.gamma_lengths = [len(ds) for ds in self.gamma_datasets]
        self.proton_lengths = [len(ds) for ds in self.proton_datasets]

        self.n_gamma = sum(self.gamma_lengths)
        self.n_proton = sum(self.proton_lengths)

    def __len__(self):
        return self.n_gamma + self.n_proton

    def _get_from_group(self, datasets, lengths, idx):
        """
        Находит, из какого датасета брать событие
        """
        cumulative = 0
        for ds, length in zip(datasets, lengths):
            if idx < cumulative + length:
                return ds.build_cnn_image(idx - cumulative)
            cumulative += length

        raise IndexError("Index out of range")

    def __getitem__(self, idx):
        if idx < self.n_gamma:
            img = self._get_from_group(self.gamma_datasets,
                                       self.gamma_lengths,
                                       idx)
            label = 0
        else:
            img = self._get_from_group(self.proton_datasets,
                                       self.proton_lengths,
                                       idx - self.n_gamma)
            label = 1

        img = torch.tensor(img).unsqueeze(0).float()
        label = torch.tensor(label).float()

        return img, label
    def show_class_balance(self):
        n_gamma = self.n_gamma
        n_proton = self.n_proton
        total = n_gamma + n_proton

        print(f"Gamma (label=0): {n_gamma}")
        print(f"Proton (label=1): {n_proton}")
        print(f"Total: {total}")