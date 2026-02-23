import argparse
import pickle
import time
import ast
import math
import csv
import os
import json
from pathlib import Path
from src.utils.utils import load_problem_hypergraph, success_probability, load_qubo_dict, evaluate_boolean_formula, tts_from_prob_and_time
#is_self_dual, check_self_dual
import neal 


def run_single_sa_simple_qubo(qubo_dict, num_reads, num_sweeps, seed=None):

    sampler = neal.SimulatedAnnealingSampler()
    kwargs = dict(
        num_reads=num_reads,
        num_sweeps=num_sweeps
    )
    if seed is not None:
        kwargs["seed"] = int(seed)
    
    t0 = time.perf_counter()
    sampleset = sampler.sample_qubo(qubo_dict, **kwargs) 
    t1 = time.perf_counter()
    return sampleset, t1 - t0







def main():
    parser = argparse.ArgumentParser(
        description="Run neal SA on QUBO directory. Success = boolean formula satisfied (no energy thresholds)."
    )
    parser.add_argument("--problem-dir", type=str, required=True,
                        help="Directory with .json hypergraph dicts.")
    parser.add_argument("--qubo-dir", type=str, required=True,
                        help="Directory with .pkl QUBO dicts.")
    parser.add_argument("--out", type=str, default="sa_boolean_results.csv",
                        help="CSV output.")
    parser.add_argument("--configs", type=str,
                        default="[(1000,2000), (1000,5000)]",
                        help="SA configurations.")
    parser.add_argument("--target-conf", type=float, default=0.99,
                        help="Target confidence for TTS.")
    args = parser.parse_args()

    problem_dir = Path(args.problem_dir)
    qubo_dir = Path(args.qubo_dir)
    out_path = Path(args.out)
    configs = parse_configs(args.configs)

    qubo_files = sorted(qubo_dir.glob("*.pkl"))
    print(f"Found {len(qubo_files)} QUBO files in {qubo_dir}")

    fieldnames = ["example_name", "config_index",
                  "num_reads", "num_sweeps", "seed",
                  "run_time_s", "best_energy_run",
                  "success_prob", "tts_target_conf",
                  "is_solution_best"]

    with out_path.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for qubo_path in qubo_files:
            example_name = qubo_path.stem
            print(f"\n=== Example: {example_name} ===")
            
            qubo_dict, formula_str = load_qubo_dict(qubo_path)

            hyperedges_list, problem_vertex_size = load_problem_hypergraph(os.path.join(problem_dir, f'{example_name}.json'))

            samplesets = []
            per_config_rows = []

            # Run all configs
            for idx, cfg in enumerate(configs):
                print(f"  Config {idx+1}/{len(configs)}: {cfg}")
                ss, wall = run_single_sa_simple_qubo(
                    qubo_dict,
                    num_reads=cfg["num_reads"],
                    num_sweeps=cfg["num_sweeps"],
                    seed=cfg["seed"],
                )
                samplesets.append(ss)
                
                # success_prob sin umbrales: cuenta reads que satisfacen fórmula
                #p = success_probability(ss, formula_str)
                p = success_probability(ss, hyperedges_list, problem_vertex_size)
                tts = tts_from_prob_and_time(p, wall, args.target_conf)
            
                
                per_config_rows.append({
                    "example_name": example_name,
                    "config_index": idx,
                    "num_reads": cfg["num_reads"],
                    "num_sweeps": cfg["num_sweeps"],
                    "seed": cfg["seed"],
                    "run_time_s": wall,
                    "success_prob": p,
                    "tts_target_conf": tts
                })
                writer.writerow(per_config_rows[-1])

    print(f"\nSaved boolean satisfaction results to {out_path}")





def parse_configs(configs_str):
    
    obj = ast.literal_eval(configs_str)
    if isinstance(obj, dict):
        obj = [obj]
    configs = []
    for item in obj:
        if isinstance(item, dict):
            cfg = {
                "num_reads": int(item.get("num_reads", 1000)),
                "num_sweeps": int(item.get("num_sweeps", 2000)),
                "seed": item.get("seed", 0),
            }
        elif isinstance(item, (list, tuple)):
            cfg = {
                "num_reads": int(item[0]),
                "num_sweeps": int(item[1]),
                "seed": item[2] if len(item) >= 3 else 0,
            }
        configs.append(cfg)
    return configs



if __name__ == "__main__":
    main()
    #python run_sa_boolean.py --problem-dir ./data/problems/hypergraphs_not_self_dual --qubo-dir ./data/QUBOs/hypergraphs_not_self_dual --out results_tts_sa.csv --configs "[(10000,5), (10000,20), (10000,100), (10000,200)]" --target-conf 0.99




