import sympy as sp
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ
from collections import defaultdict
import numpy as np
import json
import pickle
import math

class Binary(sp.Symbol):
    def __init__(self, boolean_attr):
        super()

    def _eval_power(self, expt):
        return self


def shift_variables(clauses, shift=1):

    shifted = []
    for clause in clauses:
        if shift < 0:
            new_clause = [lit + shift if lit > 0 else -(abs(lit) + shift) for lit in clause]
        else:
            new_clause = [lit + shift if lit >= 0 else -(abs(lit) + shift) for lit in clause]
        shifted.append(new_clause)
    return shifted


def get_poly_from_dimacs(dimacs, x, all_variables):

    p_i_r, *gens = ring(x, ZZ, order='lex')
    p_full = p_i_r.zero

    for clause in dimacs:

        p_i = None
        p_i = p_i_r.one

        for literal in clause:

            lit = (1 - gens[abs(literal) - 1]) if literal < 0 else gens[abs(literal) - 1]
            p_i *= lit 
        
        p_full = p_full + p_i

    return p_full.to_dict()



def split_into_low_and_high_degree_polynomial(polynomial, coeffs, degree=2): # degree -> less or equal
    
    m = polynomial.sum(axis=1) > degree
    
    return polynomial[~m], coeffs[~m], polynomial[m], coeffs[m]


def polynomial_to_QUBO(polynomial, coeffs):
    one_indices = np.argwhere(polynomial == 1)
    
    QUBO = defaultdict(float)
    
    for monom_index in range(polynomial.shape[0]):
        
        indices = np.argwhere(one_indices[:, 0] == monom_index).flatten()
        k = coeffs[monom_index]
        
        if indices.shape[0] == 1:
            i, j = one_indices[indices[0], 1], one_indices[indices[0], 1]
            QUBO[(int(i), int(j))] += float(k)
        elif indices.shape[0] == 2:
            i, j = one_indices[indices[0], 1], one_indices[indices[1], 1]
            QUBO[(int(i), int(j))] += float(k)
        elif indices.shape[0] == 0:
            offset = float(k)
        else:
            raise Exception("Not a quadratic monom")

    return QUBO

def mlb_from_monomials(monoms, coeffs, i, j):

    mask = (monoms[:, i] == 1) | (monoms[:, j] == 1)
    sum_abs_coeffs = np.abs(coeffs[mask]).sum()
    M_ij = 1.0 + 2.0 * sum_abs_coeffs 
    return float(M_ij)

def find_substitutions_and_M(monoms, coeffs, degree=2):
    
    subs = []
    monoms_matrix = monoms.copy()
    coeffs_matrix = coeffs.copy()
    M = []
    reference_monoms_matrix = monoms.copy()
    reference_coeffs = coeffs.copy()

    rows_mask = monoms_matrix.sum(axis=1) > degree
    monoms_matrix = monoms_matrix[rows_mask]
    coeffs_matrix = coeffs_matrix[rows_mask]


    while monoms_matrix.size:
        current_subs = []
        
        idx_row = np.argmax(monoms_matrix.sum(axis=1))

        cols = list(np.flatnonzero(monoms_matrix[idx_row]))
        
        while len(cols) >= 2:
            i = int(cols.pop(0))
            j = int(cols.pop(0))
        
            current_subs.append((i, j))

            m_value = mlb_from_monomials(reference_monoms_matrix, reference_coeffs, i, j)

            M.append(m_value)
                
        monoms_matrix = apply_substitutions_pairs_strict_fixed(monoms_matrix, current_subs)

        rows_mask = monoms_matrix.sum(axis=1) > degree
        monoms_matrix = monoms_matrix[rows_mask]
        coeffs_matrix = coeffs_matrix[rows_mask]
        
        subs += current_subs

    return subs, M


def apply_substitutions_pairs_strict_fixed(monoms, subs):

    M = monoms.copy()
    
    for pair_idx, (i, j) in enumerate(subs):

        mask = (M[:, i] == 1) & (M[:, j] == 1)
        rows_idx = np.nonzero(mask)[0]
        
        if rows_idx.size == 0:
            continue
            
        M[rows_idx, i] = 0
        M[rows_idx, j] = 0
    
    return M

def add_variables_to_polynomial(polynomial, n_vars_to_add):
    n_monoms, n_vars = polynomial.shape
    return np.concatenate((polynomial, np.zeros((n_monoms, n_vars_to_add), dtype=polynomial.dtype)), axis=1)


def substitute_numpy_array(X, pair, new_var_index):

    i, j = pair
    m = (X[:, i] == 1) & (X[:, j] == 1)  
    X[m, i] = 0                          
    X[m, j] = 0                          
    X[m, new_var_index] = 1              
    return X

def add_monomial(monoms, coeffs, var_indices, coeff_value):

    new_row = np.zeros((1, monoms.shape[1]), dtype=monoms.dtype)
    for idx in var_indices:
        new_row[0, idx] = 1
    monoms = np.vstack([monoms, new_row])
    coeffs = np.append(coeffs, coeff_value)
    return monoms, coeffs


def load_problem_hypergraph(path):
    
    with open(path) as f:
        json_data = json.load(f)

        problem_vertex_size = json_data['n']
        hyperedges_list = json_data['psi']
    
    return hyperedges_list, problem_vertex_size


def is_satisfying(formula, assignment, formula_type='dnf'):

    if formula_type == 'cnf':
        for clause in formula:
            clause_true = any(
                assignment[abs(lit)] == (lit > 0) 
                for lit in clause
            )
            if not clause_true:
                return False
        return True
    
    elif formula_type == 'dnf':
        for term in formula:
            term_true = all(
                assignment[abs(lit)] == (lit > 0) 
                for lit in term
            )
            if term_true:
                return True
        return False



def negate_variables(clauses):
    return [[-lit for lit in clause] for clause in clauses]

def check_self_dual(dnf_terms, assign):

    dnf_terms_shifted = shift_variables(dnf_terms, shift=1)

    f_result = is_satisfying(dnf_terms_shifted, assign, 'dnf')
    f_negated_result = is_satisfying(negate_variables(dnf_terms_shifted), assign, 'dnf')
    
    if not f_result and not f_negated_result:
        return False
    elif (f_result and not f_negated_result) or (not f_result and f_negated_result):
        return True
    else:
        raise Exception()


def num_variables(formula):
    if not formula or not any(clause for clause in formula):
        return 0
    
    max_var = max(abs(lit) for clause in formula for lit in clause)
    return max_var

def int_to_assignment(i, num_vars):

    assignment = [None]  # ignored 0 index
    
    for bit in range(num_vars):
        var_val = bool((i >> bit) & 1)
        assignment.append(var_val)
    
    return assignment

def is_self_dual(dnf_terms):
    num_vars = num_variables(dnf_terms) + 1
    for i in range(2**num_vars):
        assignment = int_to_assignment(i, num_vars)
        if not check_self_dual(dnf_terms, assignment):
            return False
    return True

def load_qubo_dict(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a QUBO dict.")
    
    formula_str = obj.get("formula_str", None) 
    qubo_dict = {k: v for k, v in obj.items() if isinstance(k, tuple)}
    return qubo_dict, formula_str


def evaluate_boolean_formula(sample, hyperedges_list, problem_vertex_size):

    assignment = [None] + [True if sample[i] == 1 else False for i in range(problem_vertex_size)]

    return is_self_dual(hyperedges_list) == check_self_dual(hyperedges_list, assignment)


def success_probability(sampleset, hyperedges_list, problem_vertex_size):

    total_reads = sampleset.record['num_occurrences'].sum()
    print("total reads", total_reads)
    
    successful_reads = 0
    
    for i in range(len(sampleset.record)):
        sample = sampleset.record['sample'][i]
        num_occ = sampleset.record['num_occurrences'][i]
        
        if evaluate_boolean_formula(sample, hyperedges_list, problem_vertex_size):
            successful_reads += num_occ
    
    return successful_reads / max(total_reads, 1)

def tts_from_prob_and_time(p, run_time, target_conf=0.99):
    if p <= 0.0:
        return float("inf")
    if p >= 1.0:
        return run_time
    return run_time * math.log(1.0 - target_conf) / math.log(1.0 - p)
