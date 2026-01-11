import sympy as sp
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ
from collections import defaultdict
import numpy as np

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
        # Máscara SOLO para ESTE par
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