from nbformat import reads
import pandas as pd
import os
import pickle
from src.utils.utils import check_not_self_dual, load_problem_hypergraph, success_probability, tts_from_prob_and_time
import dimod
import glob
import re
from pathlib import Path
from collections import defaultdict

def get_files_dictionary(base_dir):
    
    results = defaultdict(list)
    root_path = Path(base_dir)

    if not root_path.exists():
        print(f"Error: directory {base_dir} does not exist.")
        return {}

    for file_path in root_path.rglob('*.pkl'):
        if file_path.is_file():
            
            file_name = file_path.name
            #prefix =file_name.split("_cfg")[0]
            prefix =file_name.split("_reads")[0]
            results[prefix].append(str(file_path))
            
    return dict(results)


def compute_and_save_metrics():
    base_directory = "./data/results_qa/"
    file_dictionary = get_files_dictionary(base_directory)
    df_results = pd.DataFrame(columns=["Example name", "#reads", "Annealing time $(\\mu s)$", "Annealing sampling time $(s)$", "Success prob", "TTS"])

    for prefix, file_list in file_dictionary.items():
        sampleset_merged = None
        total_reads = 0
        annealing_time = 0 
        for file in file_list:
            with open(file, "rb") as f:
                serial = pickle.load(f)

            if sampleset_merged is None:
                sampleset_merged = dimod.SampleSet.from_serializable(serial)
                annealing_time = sampleset_merged.info.get('timing', {}).get('qpu_sampling_time', 0) / 1e6
                annealing_time_us = sampleset_merged.info.get('timing', {}).get('qpu_anneal_time_per_sample', 0)
            else:
                sampleset_current = dimod.SampleSet.from_serializable(serial)

                annealing_time += sampleset_current.info.get('timing', {}).get('qpu_sampling_time', 0) / 1e6
                sampleset_merged = dimod.concatenate([sampleset_merged, sampleset_current])


            
        sampleset_merged = sampleset_merged.aggregate()
        num_reads  = sampleset_merged.record['num_occurrences'].sum()

        
        
        t = sampleset_merged.info.get('timing', {})
        
        print(f"Total reads for {prefix}: {num_reads}, Number of samples: {len(sampleset_merged)}")

        
        t_run = annealing_time / num_reads

        example_name = prefix.split("_cfg")[0]
        if "2026" in example_name:
            problem_dir = os.path.join(".", "data", "problems", "hypergraphs_not_self_dual_2026_02_26") 
        else:
            problem_dir = os.path.join(".", "data", "problems", "hypergraphs_not_self_dual_2025_09_29")
        
        hyperedges_list, problem_vertex_size = load_problem_hypergraph(os.path.join(problem_dir, f'{example_name}.json'))
        success_prob = success_probability(sampleset_merged, hyperedges_list, problem_vertex_size)
        tts = tts_from_prob_and_time(success_prob, t_run, target_conf=0.99)

        result_row = {"Example name": example_name, 
                    "#reads": num_reads, 
                    "Annealing time $(\\mu s)$": int(annealing_time_us), 
                    "Annealing sampling time $(s)$": annealing_time, 
                    "Success prob": success_prob, 
                    "TTS": tts}
        df_results = pd.concat([pd.DataFrame([result_row]), df_results], axis=0,  ignore_index=True)


    
    df_results.to_csv("results_tts_qa2026_06_21_09_00_00.csv")
        
       

if __name__ == "__main__":
    compute_and_save_metrics()