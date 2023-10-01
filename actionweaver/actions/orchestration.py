class Orchestration:
    def __init__(self, data: dict = None):
        self.data = {}

        if data is not None:
            self.data |= data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())

    def items(self):
        return list(self.data.items())

    def get(self, key, default=None):
        return self.data.get(key, default)

    def clear(self):
        self.data.clear()

    def update(self, *args, **kwargs):
        self.data.update(*args, **kwargs)

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)

    def __eq__(self, other):
        if isinstance(other, Orchestration):
            return self.data == other.data
        return False
