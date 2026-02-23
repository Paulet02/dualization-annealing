import pandas as pd
import os
import pickle
from src.utils.utils import load_problem_hypergraph, success_probability, tts_from_prob_and_time
import dimod
import glob


files = glob.glob(os.path.join(".", "data", "results_qa", "hypergraphs_not_self_dual", "*"))
problem_dir = os.path.join(".", "data", "problems", "hypergraphs_not_self_dual")
df_results = pd.DataFrame(columns=["Example name", "#reads", "Annealing time $(\\mu s)$", "Annealing sampling time $(s)$", "Success prob", "TTS"])

if __name__ == "__main__":

    for file in files:

        full_name = os.path.basename(file)
        example_name, ext = os.path.splitext(full_name)

        example_name = example_name.split("_reads")[0]

        print(f'\n=== Example: {example_name} ===')

        with open(file, "rb") as f:
            serial = pickle.load(f)

        sampleset = dimod.SampleSet.from_serializable(serial)

        num_reads = sampleset.info['total_num_reads']
        annealing_time_us = sampleset.info['batch_timings'][0]['qpu_anneal_time_per_sample']

        annealing_time = sum([batch_time['qpu_sampling_time'] for batch_time in sampleset.info['batch_timings']]) / 1e6
        t_run = annealing_time / num_reads

        hyperedges_list, problem_vertex_size = load_problem_hypergraph(os.path.join(problem_dir, f'{example_name}.json'))
        success_prob = success_probability(sampleset, hyperedges_list, problem_vertex_size)
        tts = tts_from_prob_and_time(success_prob, t_run, target_conf=0.99)

        result_row = {"Example name": example_name, 
                    "#reads": num_reads, 
                    "Annealing time $(\\mu s)$": int(annealing_time_us), 
                    "Annealing sampling time $(s)$": annealing_time, 
                    "Success prob": success_prob, 
                    "TTS": tts}

        df_results = pd.concat([pd.DataFrame([result_row]), df_results], axis=0,  ignore_index=True)

    df_results.to_csv("results_tts_qa.csv")