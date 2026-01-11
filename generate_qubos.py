import os
from pathlib import Path
import argparse
import joblib
import json
from src.utils.utils import shift_variables, Binary, get_poly_from_dimacs, split_into_low_and_high_degree_polynomial, polynomial_to_QUBO, find_substitutions_and_M, add_variables_to_polynomial, substitute_numpy_array, add_monomial
import copy
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ
from collections import defaultdict
import numpy as np

def build_and_save_qubo(hypergraph, example_QUBO_path):

    example_path = hypergraph.absolute()

    with open(example_path) as f:
        json_data = json.load(f)

    problem_vertex_size = json_data['n']
    hyperedges_list = json_data['psi']

    hyperedges_list = shift_variables(hyperedges_list, shift=1)


    # f(x) + f(-x)
    hyperedges_list = hyperedges_list + [[-node for node in hyperedge] for hyperedge in hyperedges_list]

    x = tuple(Binary(f"x{i}") for i in range(problem_vertex_size))
    all_variables = [x_i for x_i in x]
    substitutions = {}

    y_vars = []

    qubo_dict = defaultdict(float)

    
    polynomial_dict = get_poly_from_dimacs(hyperedges_list, x, all_variables)
    poly_ring, *gens = ring(copy.deepcopy(all_variables), ZZ, order='lex')
    polynomial = poly_ring.from_dict(polynomial_dict)
    numpy_monoms = np.array(polynomial.monoms(), dtype=np.int8)
    numpy_coeffs = np.array(polynomial.coeffs(), dtype=np.float32)


    low_polynomial, low_coeffs, numpy_monoms, numpy_coeffs = split_into_low_and_high_degree_polynomial(numpy_monoms, numpy_coeffs, degree=1)
    
    if low_polynomial.shape[0] > 0:
        qubo_new = polynomial_to_QUBO(low_polynomial, low_coeffs)
        qubo_dict = {k: qubo_dict.get(k, 0) + qubo_new.get(k, 0) for k in set(qubo_dict) | set(qubo_new)}
    

    if numpy_monoms.shape[0] > 0:

        subs_pairs, M = find_substitutions_and_M(numpy_monoms, numpy_coeffs, degree=2)

    while len(subs_pairs) > 0:

        current_subs = {}
        numpy_monoms = add_variables_to_polynomial(numpy_monoms, len(subs_pairs))

        for ((xi_index, xj_index), M_value) in zip(subs_pairs, M):
            xi = all_variables[xi_index]
            xj = all_variables[xj_index]
            y_vars.append(Binary("y" + str(len(y_vars))))
            all_variables.append(y_vars[-1])
            current_subs.update({(xi_index, xj_index): len(all_variables) - 1})
            
            numpy_monoms = substitute_numpy_array(numpy_monoms, (xi_index, xj_index), len(all_variables) - 1)

            qubo_new = {(xi_index, xj_index): M_value, 
                        (len(all_variables)-1, len(all_variables)-1): 3*M_value}

            numpy_monoms, numpy_coeffs = add_monomial(numpy_monoms, numpy_coeffs, [xi_index, len(all_variables)-1], -2*M_value)
            numpy_monoms, numpy_coeffs = add_monomial(numpy_monoms, numpy_coeffs, [xj_index, len(all_variables)-1], -2*M_value)

            qubo_dict = {k: qubo_dict.get(k, 0) + qubo_new.get(k, 0) for k in set(qubo_dict) | set(qubo_new)}
            

        substitutions.update(current_subs)

        low_polynomial, low_coeffs, numpy_monoms, numpy_coeffs = split_into_low_and_high_degree_polynomial(numpy_monoms, numpy_coeffs, degree=1)

        if low_polynomial.shape[0] > 0:
            
            qubo_new = polynomial_to_QUBO(low_polynomial, low_coeffs)
            qubo_dict = {k: qubo_dict.get(k, 0) + qubo_new.get(k, 0) for k in set(qubo_dict) | set(qubo_new)}

        subs_pairs = []

        if numpy_monoms.shape[0] > 0:

            subs_pairs, M = find_substitutions_and_M(numpy_monoms, numpy_coeffs, degree=2)



    if numpy_coeffs.shape[0] > 0:

        qubo_new = polynomial_to_QUBO(numpy_monoms, numpy_coeffs)
        qubo_dict = {k: qubo_dict.get(k, 0) + qubo_new.get(k, 0) for k in set(qubo_dict) | set(qubo_new)}

  
    joblib.dump(qubo_dict, example_QUBO_path)


def process_hypergraphs(input_dir, qubo_dir, overwrite=False):

    input_dir = Path(input_dir)
    qubo_dir = Path(qubo_dir)

    qubo_dir.mkdir(parents=True, exist_ok=True)

    hypergraphs = list(input_dir.glob("*.json"))
    
    print(f"Found {len(hypergraphs)} hypergraph files in {input_dir}")

    for hypergraph in hypergraphs:
        example_name = hypergraph.stem  
        example_path = hypergraph.absolute()

        example_qubo_path = qubo_dir / f"{example_name}.pkl"
        
        if example_qubo_path.exists() and not overwrite:
            print(f"Skipping {example_name} (already exists)")
            continue

        print(f"Processing {example_name}...")
        
        build_and_save_qubo(example_path, example_qubo_path)

        print(f"  -> Would save QUBO to: {example_qubo_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate QUBOs from non-self-dual hypergraph JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    current_dir = os.getcwd()

    default_input = os.path.join(current_dir, "data", "problems", "hypergraphs_not_self_dual")
    default_qubo = os.path.join(current_dir, "data", "QUBOs", "hypergraphs_not_self_dual")

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=default_input,
        help="Input directory containing hypergraph JSON files"
    )
    parser.add_argument(
        "--qubo-dir", "-q",
        type=str,
        default=default_qubo,
        help="Output directory for QUBO files (.pkl)"
    )
    
    parser.add_argument(
        "--overwrite", "-w",
        action="store_true",
        help="Overwrite existing QUBO files"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    process_hypergraphs(
        input_dir=args.input_dir,
        qubo_dir=args.qubo_dir,
        overwrite=args.overwrite
    )
    
    print("Processing complete!")

    #python generate_qubos.py  --input-dir ./data/problems/hypergraphs_not_self_dual --qubo-dir ./data/QUBOs/hypergraphs_not_self_dual