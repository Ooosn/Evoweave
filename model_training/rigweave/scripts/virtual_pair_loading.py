"""Ordered, bounded CPU-only loading without advancing the consumed sampler."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time


class LoadedBatchDataset:
    def __init__(self, requests, batches):
        if len(requests) != len(batches) or len(set(requests)) != len(requests):
            raise ValueError('loaded microbatch request identities differ')
        self.batches = dict(zip(requests, batches))

    def __getitem__(self, request):
        return self.batches[request]


class OrderedAssetPrefetch:
    def __init__(self, dataset, planned_batches, *, workers, depth=2):
        if type(workers) is not int or workers < 1 or type(depth) is not int or not 1 <= depth <= 2:
            raise ValueError('CPU prefetch requires positive workers and depth1or2')
        self.dataset = dataset
        self.plan = iter(planned_batches)
        self.depth = depth
        self.pending = deque()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='pair-data')
        self.closed = False
        try:
            self._fill()
        except BaseException:
            self.close()
            raise

    def _load(self, request):
        started = time.monotonic()
        return self.dataset[request], time.monotonic() - started

    def _fill(self):
        while not self.closed and len(self.pending) < self.depth:
            try:
                requests = list(next(self.plan))
            except StopIteration:
                break
            if not requests:
                raise ValueError('empty prefetch microbatch')
            self.pending.append((requests, [self.executor.submit(self._load, r) for r in requests]))

    def take(self, requests):
        if self.closed or not self.pending:
            raise RuntimeError('prefetch plan exhausted')
        expected, futures = self.pending[0]
        if list(requests) != expected:
            raise RuntimeError('consumed requests differ from the independent prefetch plan')
        started = time.monotonic()
        values = [future.result() for future in futures]
        wait = time.monotonic() - started
        self.pending.popleft()
        self._fill()
        return LoadedBatchDataset(expected, [value[0] for value in values]), {
            'cpu_data_wait_seconds': wait,
            'cpu_data_task_seconds': sum(value[1] for value in values)}

    def close(self):
        if not self.closed:
            self.closed = True
            for _, futures in self.pending:
                for future in futures:
                    future.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
