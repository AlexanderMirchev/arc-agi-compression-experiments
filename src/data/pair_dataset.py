class PairDataset:
    def __init__(self, pairs, preprocess_fn, compress_fn=None, process_output=True):
        a,b = zip(*pairs)
        self.a = a
        self.b = b
        self.preprocess_fn = preprocess_fn
        self.compress_fn = compress_fn
        self.process_output = process_output

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        a = self.a[idx]
        b = self.b[idx]

        a = self.preprocess_fn(a)
        if self.process_output:
            b = self.preprocess_fn(b)

        if self.compress_fn:
            a = self.compress_fn(a)
            if self.process_output:
                b = self.compress_fn(b)

        return a, b