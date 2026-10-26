from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import unittest
import torch
from tmp.train_mlm_objective_pilot_20261024 import initialize_worker


def thread_counts(_):
    return torch.get_num_threads(),torch.get_num_interop_threads()


class ReusedWorkerChecks(unittest.TestCase):
    def test_one_initializer_multiple_tasks(self):
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
            self.assertEqual(list(pool.map(thread_counts,range(3))),[(1,1)]*3)


if __name__=='__main__':unittest.main()
