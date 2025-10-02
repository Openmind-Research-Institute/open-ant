
def safe_json(o):
    if isinstance(o, type):          # classes, e.g. <class 'float'>
        return str(o)
    if hasattr(o, "dtype"):          # torch/numpy dtypes
        return str(o)
    if hasattr(o, "tolist"):         # numpy arrays
        return o.tolist()
    if hasattr(o, "__dict__"):       # custom objects
        return o.__dict__
    return str(o)
