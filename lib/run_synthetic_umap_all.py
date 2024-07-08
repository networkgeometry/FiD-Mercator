import glob
import sys
import os
import random


def embed_network(edgelist_path, umap_coords_path, is_slow) -> int:
    flag = '' if is_slow else '-f' 
    seed = random.randint(0, 999999)
    command = f"./mercator -s {seed} -d 2 {flag} -c -v -u {umap_coords_path} {edgelist_path}"
    os.system(command)


if __name__ == '__main__':
    os.system("./build.sh -b Release")
    folder = sys.argv[1]
    for coords in sorted(glob.glob(f"{folder}/*/*/*.inf_coord")):
        edgelist = f"{coords.split('.')[0]}.edge"

        # 1. Create synthetic features
        create_features_command = f"python lib/craft_synthetic_features_umap_spherical.py {edgelist}"
        os.system(create_features_command)
        local_features_filename = f"{edgelist.split('.')[0]}.pos_umap_local_features"
        global_features_filename = f"{edgelist.split('.')[0]}.pos_umap_global_features"

        # 2. Create directories and copy edgelists
        folder = os.path.dirname(edgelist)
        umap_ml_local_dir = f"{folder}/umap_ml_local"
        umap_ml_global_dir = f"{folder}/umap_ml_global"
        only_umap_local_dir = f"{folder}/only_umap_local"
        only_umap_global_dir = f"{folder}/only_umap_global"

        os.system(f"mkdir {umap_ml_local_dir} {umap_ml_global_dir} {only_umap_local_dir} {only_umap_global_dir}")
        os.system(f"cp {edgelist} {umap_ml_local_dir}")
        os.system(f"cp {edgelist} {umap_ml_global_dir}")
        os.system(f"cp {edgelist} {only_umap_local_dir}")
        os.system(f"cp {edgelist} {only_umap_global_dir}")

        # 3. Embed networks
        edgelist_filename = os.path.basename(edgelist)
        embed_network(f"{umap_ml_local_dir}/{edgelist_filename}", local_features_filename, True)
        embed_network(f"{umap_ml_global_dir}/{edgelist_filename}", global_features_filename, True)
        embed_network(f"{only_umap_local_dir}/{edgelist_filename}", local_features_filename, False)
        embed_network(f"{only_umap_global_dir}/{edgelist_filename}", global_features_filename, False)