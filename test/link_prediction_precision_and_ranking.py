import os
import sys
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    d = sys.argv[1]
    inf_coords_path = sys.argv[2]
    edgelist_path = sys.argv[3]

    beta, mu = -1, -1
    with open(inf_coords_path, 'r') as f:
        for line in f:
            if 'beta:' in line:
                beta = line.split()[-1]
            elif 'mu:' in line:
                mu = line.split()[-1]

    fraction_missing_links = np.linspace(0.01, 0.99, num=50)

    for q in tqdm(fraction_missing_links):
        q_str = "{:.2f}".format(q)
        results_filename = f"{os.path.dirname(edgelist_path)}/{os.path.split(edgelist_path)[-1].split('.')[0]}_q={q_str.replace('.', '_')}.lp"
        command = f"""
            g++ --std=c++17 -O3 lib/link_prediction.cpp -o lp2.out && ./lp2.out {d} {beta} {mu} {inf_coords_path} {edgelist_path} 10 {q_str} > {results_filename}
        """
        os.system(command)

