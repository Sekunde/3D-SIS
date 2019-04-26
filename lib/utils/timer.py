import time
import torch

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self._total_time = 0
        self._calls = 0
        self._start_time = 0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        torch.cuda.synchronize()
        self._start_time = time.time()

    def toc(self):
        torch.cuda.synchronize()
        _diff = time.time() - self._start_time
        self._total_time = self._total_time + _diff
        self._calls += + 1

    def average_time(self, name='default'):
        return self._total_time / self._calls

    def total_time(self, name='default'):
        return self._total_time

    def clean_total_time(self):
        self._total_time = 0
        self._calls = 0

timer = Timer()
