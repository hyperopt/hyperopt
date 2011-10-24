import base
from gdist import gDist


class GensonBandit(base.Bandit):
    def __init__(self, source_file=None, source_string=None):
        if source_file is not None:
            if isinstance(source_file, str):
                source_file = open(source_file)
            source_string = source_file.read()
        base.Bandit.__init__(self, gDist(source_string))
