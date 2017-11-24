import zmq


class Context(object):

    _default_context = None

    @staticmethod
    def default_context():
        if Context._default_context is None:
            Context._default_context = zmq.Context()
        return Context._default_context


class Connection(object):

    def __init__(self, address, sock_type, context=None, linger=0):
        if context is None:
            context = Context.default_context()
        self._context = context
        self._sock_type = sock_type
        self._address = address
        self._socket = None
        self._connected = False
        self._bound = False
        self._linger = linger

    @property
    def zmq_socket(self):
        return self._socket

    def is_connected(self):
        return self._connected

    def is_bound(self):
        return self._bound

    def create_socket(self):
        self._socket = self._context.socket(self._sock_type)
        self._socket.setsockopt(zmq.LINGER, self._linger)

    def bind(self):
        if self._socket is None:
            self.create_socket()
        self._socket.bind(self._address)
        self._bound = True

    def unbind(self):
        self._socket.unbind(self._address)
        self._bound = False

    def connect(self):
        if self._socket is None:
            self.create_socket()
        # print("Connecting to {}".format(self._address))
        self._socket.connect(self._address)
        self._connected = True

    def disconnect(self):
        self._socket.disconnect(self._address)
        self._connected = False

    def reconnect(self):
        self.disconnect()
        self.destroy_socket()
        self.create_socket()
        self.connect()

    def destroy_socket(self):
        self._socket.close()
        self._socket = None
        self._connected = False

    def poll(self, timeout=None, flags=zmq.POLLIN):
        return self._socket.poll(timeout, flags)

    def send(self, data, flags=0, copy=True, track=False):
        return self._socket.send(data, flags, copy, track)

    def recv(self, flags=0, copy=True, track=False, timeout=None):
        if timeout is None:
            return self._socket.recv(flags, copy, track)
        else:
            result = self.poll(timeout, zmq.POLLIN)
            if result == zmq.POLLIN:
                return self._socket.recv(flags, copy, track)
            else:
                return None

    def send_recv(self, data, flags_out=0, flags_in=0, copy_in=True, copy_out=True,
                  track_in=False, track_out=False, timeout=None):
        self.send(data, flags_out, copy_out, track_out)
        self.recv(flags_in, copy_in, track_in, timeout)