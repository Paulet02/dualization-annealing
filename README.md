# QUBO Generator (Hypergraphs)

This script generates QUBO instances from **hypergraph** JSON files. It scans an input directory for hypergraph `.json` files and writes the generated QUBOs as `.pkl` files into an output directory.


## Default directories

By default, the script uses the **current working directory** as the base path:

- Input: `./data/problems/hypergraphs_not_self_dual`
- Output: `./data/QUBOs/hypergraphs_not_self_dual`

So if you run the script from the repository root, you usually don’t need to pass any arguments.

## Usage

Run:

```bash
python generate_qubos.py
```

Or explicitly specify input/output directories:

```bash
python generate_qubos.py \
  --input-dir ./data/problems/hypergraphs_not_self_dual \
  --qubo-dir ./data/QUBOs/hypergraphs_not_self_dual
```

## Command-line arguments

- `--input-dir`, `-i`  
  Path to the directory containing hypergraph JSON files.  
  Default: `./data/problems/hypergraphs_not_self_dual`

- `--qubo-dir`, `-q`  
  Path to the directory where QUBO `.pkl` files will be written.  
  Default: `./data/QUBOs/hypergraphs_not_self_dual`

- `--overwrite`, `-w`  
  If set, overwrites existing QUBO files in the output directory.  
  Default: off (existing outputs are preserved)

## Examples

Generate QUBOs using default paths:

```bash
python generate_qubos.py
```

Generate QUBOs with custom folders:

```bash
python generate_qubos.py -i ./my_hypergraphs -q ./my_qubos
```

Regenerate everything (overwrite existing `.pkl` files):

```bash
python generate_qubos.py -w
```

## Output

When finished, the script prints:

```text
Processing complete!
```

Generated QUBO files are stored in the output directory as `.pkl`.

***
