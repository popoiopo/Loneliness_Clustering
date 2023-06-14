from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import current_process
import logging
import time

def test(a):
    time.sleep(2)
    logging.info(f'Node {a} on process {current_process().name}')
    return f"slept for {a**a} seconds"


if __name__ == "__main__":
    logging.basicConfig(filename='testing.log',
                        encoding='utf-8', level=logging.DEBUG)

    tic = time.perf_counter()
    a_s = list(range(32))
    pool = Pool(16)
    t = pool.imap_unordered(test, a_s)
    for result in t:
        logging.info(result)
    toc = time.perf_counter()
    logging.info(f"Ran code in {toc - tic:0.4f} seconds")