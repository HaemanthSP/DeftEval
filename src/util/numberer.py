class Numberer:
    def __init__(self, vocabulary):
        self.v2n = dict()
        self.n2v = list()
        self.INVALID_NUMBER = -1

        for item in vocabulary:
            _ = self.number(item)

    def number(self, value, add_if_absent=True):
        n = self.v2n.get(value)

        if n is None:
            if add_if_absent:
                n = len(self.n2v)
                self.v2n[value] = n
                self.n2v.append(value)
            else:
                n = self.INVALID_NUMBER

        return n

    def value(self, number):
        assert number > self.INVALID_NUMBER
        return self.n2v[number]

    def max_number(self):
        return len(self.n2v)
