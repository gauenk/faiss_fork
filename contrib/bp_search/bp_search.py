"""
Belief Propogation Search

"""

from .bp_search_rand import runBpSearch_rand
from .bp_with_warp import runBpSearch as runBpSearch_cluster
from .bp_search_cluster_approx import runBpSearchClusterApprox

def return_optional(pydict,key,default):
    if key in pydict: return pydict[key]
    else: return default

def delete_optional(pydict,key):
    if key in pydict: del pydict[key]

def runBpSearch(*args,**kwargs):
    search_type = return_optional(kwargs,'search_type','rand')
    delete_optional(kwargs,'search_type')
    if search_type == "rand":
        return runBpSearch_rand(*args,**kwargs)
    elif search_type == "cluster":
        return runBpSearch_cluster(*args,**kwargs)
    elif search_type == "cluster_approx":
        return runBpSearchClusterApprox(*args,**kwargs)
    else:
        raise ValueError(f"Uknown bp search [{search_type}]")
    
