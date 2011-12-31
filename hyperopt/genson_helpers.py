def string(s):
    return repr(s).replace("'",'"')

##Plain vanilla params  -- from FG11 paper

class Null(object):
    def __repr__(self):
        return 'null'
null = Null()

class FALSE(object):
    def __repr__(self):
        return 'false'
false = FALSE()

class TRUE(object):
    def __repr__(self):
        return 'true'
true = TRUE()


def repr(x):
    if isinstance(x,str):
        return '"' + str(x) + '"'
    else:
        return x.__repr__()

class gObj(object):
    def __init__(self,*args,**kwargs):
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        argstr = ', '.join([repr(x) for x in self.args])
        kwargstr = ', '.join([str(k) + '=' + repr(v) for k,v in self.kwargs.items()])

        astr = argstr + (', ' + kwargstr if kwargstr else '')
        return self.name + '(' + astr + ')'


class choice(gObj):
    name = 'choice'

class uniform(gObj):
    name = 'uniform'

class gaussian(gObj):
    name = 'gaussian'

class lognormal(gObj):
    name = 'lognormal'

class qlognormal(gObj):
    name = 'qlognormal'

class ref(object):
    def __init__(self,*p):
        self.path = p

    def __repr__(self):
        return '.'.join(self.path)

