import os
import argparse
import random
import subprocess
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    1. Input: umap coordinates + edgelist
    2. Embed using D-Mercator
        b) UMAP+ML (using umap coordinates)
        c) UMAP (only umap coordinates)

    3. Run greedy routing for all of embeddings
    """))
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to edgelist")
    parser.add_argument('-u', '--umap_coordinates', type=str, required=True, help="Path to UMAP coordinates")
    parser.add_argument('-q', '--n_runs', type=int,
                        required=False, default=100000, help="Number of greedy routing iterations.")
    args = parser.parse_args()
    return args


def embed_network(edgelist_path, umap_coordinates, flag) -> int:
    seed = random.randint(0, 999999)
    fast_flag = '-f' if flag == 'fast' else ''
    status = subprocess.Popen(
        f"./mercator -s {seed} -d 2 -c {fast_flag} -u {umap_coordinates} -v {edgelist_path}; echo $?", shell=True, stdout=subprocess.PIPE)
    exit_code = status.stdout.read().decode()
    return int(exit_code)


def run_embedding(output_folder, edgelist_path, umap_coordinates, flag):
    cp_command = f"cp {edgelist_path} {output_folder}"
    os.system(cp_command)

    output_filename = f"{output_folder}/{os.path.split(edgelist_path)[-1]}"
    embed_network(output_filename, umap_coordinates, flag)
    
    coords_path = os.path.split(output_filename)[-1].split(".")[0]
    coords_path = f"{output_folder}/{coords_path}.inf_coord"
    return output_filename, coords_path


if __name__ == '__main__':
    args = parse_args()
    edgelist = args.input
    umap_coordinates = args.umap_coordinates

    folder = os.path.dirname(umap_coordinates)
    os.makedirs(f"{folder}/umap_ml/", exist_ok=True)
    os.makedirs(f"{folder}/only_umap/", exist_ok=True)
    
    # 1. Embed UMAP+ML
    new_edgelist_path, new_coords_path = run_embedding(f'{folder}/umap_ml/', edgelist, umap_coordinates, 'slow')
    # 2  Embed UMAP (only)
    new_edgelist_path, new_coords_path = run_embedding(f'{folder}/only_umap/', edgelist, umap_coordinates, 'fast')
    