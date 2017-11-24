import six


class Serializer(object):

    def dumps(self, obj):
        raise NotImplementedError()

    def loads(self, dump):
        raise NotImplementedError()

    def dump(self, obj):
        raise NotImplementedError()

    def load(self, dump):
        raise NotImplementedError()

    def _dump_wrapper(self, obj, file):
        if isinstance(file, six.string_types):
            with open(file, "wb") as file_obj:
                return self._dump(obj, file_obj)
        self._dump(obj, file)

    def _load_wrapper(self, file):
        if isinstance(file, six.string_types):
            with open(file, "rb") as file_obj:
                return self._load(file_obj)
        return self._load(file)


class DummySerializer(Serializer):

    def dumps(self, obj):
        return obj

    def loads(self, dump):
        return dump

    def dump(self, obj):
        raise RuntimeError("Dumping to file cannot be used with DummySerializer.")

    def load(self, dump):
        raise RuntimeError("Loading from file cannot be used with DummySerializer.")


class MarshalSerializer(Serializer):

    def __init__(self, version=None):
        import marshal
        if version is None:
            version = marshal.version
        self._marshal = marshal
        self._version = version
        self._dump = marshal.dump
        self._load = marshal.load

    def dumps(self, obj):
        return self._marshal.dumps(obj, self._version)

    def loads(self, dump):
        return self._marshal.loads(dump)

    def dump(self, obj, file):
        self._dump_wrapper(obj, file)

    def load(self, file):
        return self._load_wrapper(file)


class PickleSerializer(Serializer):

    def __init__(self, protocol=None):
        import pickle
        if protocol is None:
            protocol = pickle.HIGHEST_PROTOCOL
        self._pickle = pickle
        self._protocol = protocol
        self._dump = pickle.dump
        self._load = pickle.load

    def dumps(self, obj):
        return self._pickle.dumps(obj, self._protocol)

    def loads(self, dump):
        return self._pickle.loads(dump)

    def dump(self, obj, file):
        self._dump_wrapper(obj, file)

    def load(self, file):
        return self._load_wrapper(file)


class JsonSerializer(Serializer):

    def __init__(self):
        import json
        self._json = json
        self._dump = json.dump
        self._load = json.load

    def dumps(self, obj):
        return self._json.dumps(obj)

    def loads(self, dump):
        return self._json.loads(dump)

    def dump(self, obj, file):
        self._dump_wrapper(obj, file)

    def load(self, file):
        return self._load_wrapper(file)


class UJsonSerializer(Serializer):

    def __init__(self):
        import ujson
        self._ujson = ujson
        self._dump = ujson.dump
        self._load = ujson.load

    def dumps(self, obj):
        return self._ujson.dumps(obj)

    def loads(self, dump):
        return self._ujson.loads(dump)

    def dump(self, obj, file):
        self._dump_wrapper(obj, file)

    def load(self, file):
        return self._load_wrapper(file)


class RapidJsonSerializer(Serializer):

    def __init__(self):
        import rapidjson
        self._rapidjson = rapidjson
        self._dump = rapidjson.dump
        self._load = rapidjson.load

    def dumps(self, obj):
        return self._rapidjson.dumps(obj)

    def loads(self, dump):
        return self._rapidjson.loads(dump)

    def dump(self, obj, file):
        self._dump_wrapper(obj, file)

    def load(self, file):
        return self._load_wrapper(file)


class MsgPackSerializer(Serializer):

    def __init__(self):
        from . import msgpack_utils
        self._dumps = msgpack_utils.dumps
        self._loads = msgpack_utils.loads
        self._dump = msgpack_utils.dump
        self._load = msgpack_utils.load

    def dumps(self, obj):
        return self._dumps(obj)

    def loads(self, dump):
        return self._loads(dump)

    def dump(self, obj, file):
        self._dump_wrapper(obj, file)

    def load(self, file):
        return self._load_wrapper(file)


def get_serializer_by_name(name):
    if name is None or name == "pickle":
        serializer = PickleSerializer()
    elif name == "dummy":
        serializer = DummySerializer()
    elif name == "msgpack":
        serializer = MsgPackSerializer()
    else:
        raise RuntimeError("Unsupported serialization type: {}".format(name))
    return serializer
