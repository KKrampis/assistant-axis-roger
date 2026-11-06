"""Read selected tensors from a legacy-format (non-zip) torch.save file without
loading the rest: parse the pickle header (at the start of the file), then seek
to each wanted storage.  Read only; never writes to the source file."""
import pickle, struct
import torch

ELEM = {"BFloat16Storage": (2, torch.bfloat16), "HalfStorage": (2, torch.float16),
        "FloatStorage": (4, torch.float32), "DoubleStorage": (8, torch.float64)}


class _T:
    __slots__ = ("key", "offset", "size", "stride")

    def __init__(self, key, offset, size, stride):
        self.key, self.offset, self.size, self.stride = key, offset, tuple(size), tuple(stride)


def scan(path, verify="all"):
    """``(tensors, storages, file)``: tensors maps each top-level key to a _T (or
    the plain object for non-tensors); storages maps storage key -> (data offset,
    numel, element size, dtype)."""
    f = open(path, "rb")
    pickle.load(f); pickle.load(f); pickle.load(f)     # magic, protocol, sys info
    meta = {}

    class U(pickle.Unpickler):
        def persistent_load(self, saved_id):
            assert saved_id[0] == "storage", saved_id[0]
            storage_type, root_key, _location, numel, view_metadata = saved_id[1:]
            assert view_metadata is None
            name = getattr(storage_type, "__name__", str(storage_type))
            meta[root_key] = (name, numel)
            return root_key

        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return lambda storage, offset, size, stride, *a, **k: _T(storage, offset, size, stride)
            return super().find_class(mod, name)

    obj = U(f).load()
    keys = pickle.load(f)
    pos = f.tell()
    storages = {}
    for i, key in enumerate(keys):
        name, numel = meta[key]
        elem, dtype = ELEM[name]
        if verify == "all" or i == 0:   # each record is an 8-byte count, then the data
            f.seek(pos)
            n = struct.unpack("<q", f.read(8))[0]
            assert n == numel, (key, n, numel)
        storages[key] = (pos + 8, numel, elem, dtype)
        pos += 8 + numel * elem
    return obj, storages, f, pos


def read_tensor(f, storages, t):
    off, numel, elem, dtype = storages[t.key]
    n = 1
    for s in t.size:
        n *= s
    f.seek(off + t.offset * elem)
    buf = bytearray(f.read(n * elem))
    x = torch.frombuffer(buf, dtype=dtype)
    return torch.as_strided(x, t.size, t.stride) if t.stride != _contig(t.size) else x.view(t.size)


def _contig(size):
    st, acc = [], 1
    for s in reversed(size):
        st.append(acc)
        acc *= s
    return tuple(reversed(st))
