import numpy as np
import torch

joints_left = [4, 5, 6, 11, 12, 13]
joints_right = [1, 2, 3, 14, 15, 16]

class data_prefetcher():
    def __init__(self, loader, device, is_train=False):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.is_train = is_train

        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_batch.items():   # iterate over each object returned by __getitem__
                if isinstance(v, torch.Tensor):
                    self.next_batch[k] = v.cuda(non_blocking=True).to(self.device)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch
