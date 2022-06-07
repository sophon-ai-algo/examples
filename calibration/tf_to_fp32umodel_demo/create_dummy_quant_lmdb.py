import ufwio
import numpy as np

path = "./dummy_lmdb"
txn = ufwio.LMDB_Dataset(path)
for i in range(10):
    dummydata= np.random.rand(1, 299, 299, 3).astype(np.float32)*127.0
    txn.put(dummydata)
txn.close()
