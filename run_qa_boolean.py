import argparse
import pickle
import ast
from pathlib import Path

import dimod
from dwave.system import DWaveSampler, EmbeddingComposite


def parse_configs(configs_str):
    # configs like: "[(1000,20.0), (5000,50.0)]"  -> (num_reads, annealing_time_us)
    obj = ast.literal_eval(configs_str)
    return [(int(x[0]), float(x[1])) for x in obj]


def main():
    p = argparse.ArgumentParser("Run D-Wave QPU on a directory of QUBO dict pickles and save raw SampleSets.")
    p.add_argument("--qubo-dir", required=True, type=str, help="Directory with .pkl files containing QUBO dicts.")
    p.add_argument("--samples-dir", required=True, type=str, help="Where to save SampleSet pickles.")
    p.add_argument("--out-index", default="qpu_samplesets_index.csv", type=str, help="CSV with paths to saved SampleSets.")
    p.add_argument("--configs", default="[(1000,20.0)]", type=str,
                   help="List of (num_reads, annealing_time_us). Example: \"[(1000,20.0),(5000,50.0)]\"")
    p.add_argument("--profile", default=None, type=str, help="Optional D-Wave config profile name.")
    p.add_argument("--return-embedding", action="store_true", help="Store embedding context in SampleSet.info.")
    args = p.parse_args()

    qubo_dir = Path(args.qubo_dir)
    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    configs = parse_configs(args.configs)

    qpu = DWaveSampler() if args.profile is None else DWaveSampler(profile=args.profile)
    sampler = EmbeddingComposite(qpu)

    qubo_files = sorted(qubo_dir.glob("*.pkl"))

    with open(args.out_index, "w") as fcsv:
        fcsv.write("example_name,config_index,num_reads,annealing_time_us,sampleset_pickle\n")

        for qubo_path in qubo_files:
            example_name = qubo_path.stem

            with open(qubo_path, "rb") as f:
                qubo = pickle.load(f)
            if not isinstance(qubo, dict):
                raise ValueError(f"{qubo_path} must contain a QUBO dict.")

            # keep only QUBO entries (i,j)->value in case pickle contains extra keys
            qubo = {k: v for k, v in qubo.items() if isinstance(k, tuple)}

            for cfg_idx, (num_reads, annealing_time) in enumerate(configs):
                sampleset = sampler.sample_qubo(
                    qubo,
                    num_reads=num_reads,
                    annealing_time=annealing_time,
                    answer_mode="raw",
                    return_embedding=args.return_embedding
                )

                out_name = f"{example_name}_cfg{cfg_idx}_reads{num_reads}_t{annealing_time}us.pkl"
                out_path = samples_dir / out_name

                # robust save
                serial = sampleset.to_serializable()
                with open(out_path, "wb") as fout:
                    pickle.dump(serial, fout)

                fcsv.write(f"{example_name},{cfg_idx},{num_reads},{annealing_time},{out_path}\n")


if __name__ == "__main__":
    main()



#python run_qa_boolean.py --qubo-dir ./data/QUBOs/hypergraphs_not_self_dual --samples-dir ./data/results_qa/hypergraphs_not_self_dual --configs "[(10000,5.0),(10000,20.0),(10000,100.0),(10000,200.0)]"
