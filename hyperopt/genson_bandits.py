import base
from gdist import gDist


class GensonBandit(base.Bandit):
    def __init__(self, genson_file):
        template = gDist(open(genson_file).read())
        base.Bandit.__init__(self.template)
