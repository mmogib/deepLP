from typing import List
import numpy as np

def add(x:int, y:List[int]):
    np.array(y) + x

if __name__ == "__main__":
    s = add(1, [2,4])
    print(s)