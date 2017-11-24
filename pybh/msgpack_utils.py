import msgpack
import msgpack_numpy

msgpack_numpy.patch()

dumps = msgpack.dumps
loads = msgpack.loads

dump = msgpack.dump
load = msgpack.load

pack = msgpack.pack
unpack = msgpack.unpack

packb = msgpack.packb
unpackb = msgpack.unpackb
