"""
Belief Propogation Search

"""

from .bp_search_rand import runBpSearch_rand


def runBpSearch(*args,**kwargs):
    return runBpSearch_rand(*args,**kwargs)
