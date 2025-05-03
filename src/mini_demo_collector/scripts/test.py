import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import numpy as np

env = lmdb.open("realworld_dataset/lmdb", readonly=True, lock=False)
txn = env.begin()
cursor = txn.cursor()

for key, value in cursor:
    print(f"Key: {key}")
    data = msgpack.unpackb(value, object_hook=msgpack_numpy.decode)
    print("Loaded data keys:", data.keys())
    for k, v in data.items():
        print(f"  {k}: {type(v)} shape={np.shape(v)}")
    break  # Only inspect first one
