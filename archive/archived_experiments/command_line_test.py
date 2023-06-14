import sys
import time
from multiprocessing.pool import ThreadPool as Pool


def test(a):
    points = [[float(f) for f in p.split(",")] for p in sys.argv[1:]]
    print(a, points)
    time.sleep(10)
    return f"Done with {a=}!"


if __name__ == "__main__":
    a_s = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    with Pool(processes=len(a_s)) as pool:
        results = pool.imap_unordered(test, a_s)
        for result in results:
            print(result)
