
from nnf_utils import runNnfBurst as runL2LocalSearch
from bnnf_utils import runBurstNnf as runBurstL2LocalSearch
from bp_search import runBpSearch

def align_interface(method_name):
    if method_name == "pair_l2_local":
        return runL2LocalSearch
    elif method_name == "exh_jointly_l2_local":
        return runBurstL2LocalSearch
    elif method_name == "bp_jointly_l2_local":
        return runBpSearch
    else:
        raise KeyError(f"Uknown faiss alignment method {method_name}")
