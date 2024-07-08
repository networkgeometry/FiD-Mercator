from operator import itemgetter
from numba import jit
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import math

COLORS = [
    "#a1483f",
    "#6bb541",
    "#9c5bce",
    "#c6af38",
    "#5869cc",
    "#d4802a",
    "#5d92cd",
    "#d34b34",
    "#4cbbb2",
    "#ca50b2",
    "#58bc7f",
    "#d54175",
    "#437d42",
    "#a984ce",
    "#a3ac5d",
    "#964a7a",
    "#786d26",
    "#de86b7",
    "#c58953",
    "#e17f78"
]

@jit(nopython=True)
def euclidean_to_hyperspherical_coordinates(vec):
    # From: https://en.wikipedia.org/wiki/N-sphere
    # vec -- coordinates of node with size D+1
    r = np.linalg.norm(vec)
    angles = [r]
    for i in range(len(vec) - 2):
        bottom = 0
        for j in range(i, len(vec)):
            bottom += vec[j] * vec[j]
        bottom = np.sqrt(bottom)
        angles.append(np.arccos(vec[i] / bottom))

    denominator = np.sqrt(vec[-1] * vec[-1] + vec[-2] * vec[-2])
    if denominator < 1e-15:
        theta = 0
    else:
        theta = np.arccos(vec[-2] / denominator)
    if vec[-1] < 0:
        theta = 2 * np.pi - theta

    angles.append(theta)
    return angles


@jit(nopython=True)
def hyperspherical_to_euclidean_coordinates(v):
    positions = []
    angles = v[1:]
    r = v[0]
    for i in range(len(angles)):
        val = np.cos(angles[i])
        for j in range(i):
            val *= np.sin(angles[j])
        positions.append(r * val)

        if i == (len(angles) - 1):
            val = np.sin(angles[i])
            for j in range(i):
                val *= np.sin(angles[j])
            positions.append(r * val)
    return positions


@jit(nopython=True)
def compute_angular_distances(x, y):
    angular_distances = []
    for v, u in zip(x, y):
        angular_distances.append(
            np.arccos(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u))))
    return angular_distances


def get_rotation_matrix(vec, axis=np.array([1, 0, 0])):
    # From: https://math.stackexchange.com/a/476311
    a = vec / np.linalg.norm(vec)
    b = axis
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = np.eye(3) + vx + np.dot(vx, vx) * (1-c)/(s**2)
    return np.matrix(r)


def rotation_along_Xaxis(vec, theta):
    m = np.matrix([[1, 0, 0],
                   [0, np.cos(theta), np.sin(theta)],
                   [0, -np.sin(theta), np.cos(theta)]])
    return (m @ vec).A1


def rotation_matrix_XY(theta):
    m = np.matrix([[1, 0, 0],
                   [0, np.cos(theta), np.sin(theta)],
                   [0, -np.sin(theta), np.cos(theta)]])
    return m


def rotation_matrix_Z(theta):
    m = np.matrix([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])
    return m


def rotation_matrix_XYZ(gamma, beta, alpha):
    X11 = np.cos(alpha) * np.cos(beta)
    X12 = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    X13 = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    
    X21 = np.sin(alpha) * np.cos(beta)
    X22 = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    X23 = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)

    X31 = -np.sin(beta)
    X32 = np.cos(beta) * np.sin(gamma)
    X33 = np.cos(beta) * np.cos(gamma)
    
    m = np.matrix([[X11, X12, X13],
                   [X21, X22, X23],
                   [X31, X32, X33]])
    return m



def apply_pipeline_matrix(pos1, pos2, theta_num):
    mean_distances = []
    pos1_rotation_matrices = []
    pos2_rotation_matrices = []
    thetas = []

    for i in tqdm(range(len(pos1))):
        node_i = pos1[i]
        node_i_prime = pos2[i]

        m_i = get_rotation_matrix(node_i)
        m_j = get_rotation_matrix(node_i_prime)

        pos1_axis = np.array([(m_i @ v).A1 for v in pos1])
        pos2_axis = np.array([(m_j @ v).A1 for v in pos2])
        
        for theta in np.linspace(0, 2*np.pi, num=theta_num):
            # rotate only the second coordinates
            pos2_axis_theta = np.matmul(rotation_matrix_XY(theta), pos2_axis.transpose()).T

            mean_distance = compute_angular_distances(pos1_axis, pos2_axis_theta)
            mean_distances.append(np.mean(mean_distance))

            pos1_rotation_matrices.append(m_i)
            pos2_rotation_matrices.append(m_j)
            thetas.append(theta)

    min_distance_idx = np.argmin(np.array(mean_distances))
    min_pos1_rotation_matrix = pos1_rotation_matrices[min_distance_idx]
    min_pos2_rotation_matrix = pos2_rotation_matrices[min_distance_idx]
    min_theta = thetas[min_distance_idx]

    out_pos1 = np.array([(min_pos1_rotation_matrix @ v).A1 for v in pos1])
    out_pos2 = np.array([(min_pos2_rotation_matrix @ v).A1 for v in pos2])
    out_pos2 = np.matmul(rotation_matrix_XY(min_theta), out_pos2.transpose()).T
    out_pos2 = np.array(out_pos2)

    out_pos1_spherical = np.array([euclidean_to_hyperspherical_coordinates(v) for v in out_pos1])
    out_pos2_spherical = np.array([euclidean_to_hyperspherical_coordinates(v) for v in out_pos2])

    rotation_dict = {'min_pos1_rotation_matrix': min_pos1_rotation_matrix, 
                     'min_pos2_rotation_matrix': min_pos2_rotation_matrix,
                     'min_theta': min_theta}
    return out_pos1, out_pos2, out_pos1_spherical, out_pos2_spherical, rotation_dict


def normalize_coordinates(inf_coords_1, inf_coords_2):
    inf_coords_1_values = inf_coords_1[['p1', 'p2', 'p3']].values
    inf_coords_2_values = inf_coords_2[['p1', 'p2', 'p3']].values
    inf_coords_1_values /= np.linalg.norm(inf_coords_1_values, axis=1)[:, None]
    inf_coords_2_values /= np.linalg.norm(inf_coords_2_values, axis=1)[:, None]
    return inf_coords_1_values, inf_coords_2_values


def apply_pipeline_matrix_maximize_pearson(inf_coords_1, inf_coords_2, theta_num=50):
    inf_coords_1_values, inf_coords_2_values = normalize_coordinates(inf_coords_1, inf_coords_2)
    (inf_coords_1_values, best_inf_coords_euclidean, real_coords_spherical, best_inf_coords_spherical, rotation_dict) = \
        apply_pipeline_matrix(inf_coords_1_values, inf_coords_2_values, theta_num)

    # Rotate on Z-axis and find the angle corresponding to the maximum value of pearson correlation for the second angle (phi2)
    all_thetas = np.linspace(0, 2*np.pi, 50)
    all_pearson_phi1 = []
    all_pearson_phi2 = []

    for theta_z in all_thetas:
        rotate_best_inf_coords_euclidean = np.matmul(rotation_matrix_XY(theta_z), 
                                                     best_inf_coords_euclidean.transpose()).T
        rotate_best_inf_coords_euclidean = np.array(rotate_best_inf_coords_euclidean)

        rotate_best_inf_coords_spherical = np.array(
            [euclidean_to_hyperspherical_coordinates(v) for v in rotate_best_inf_coords_euclidean])

        x = real_coords_spherical[:, 1]
        y = rotate_best_inf_coords_spherical[:, 1]
        pearson_phi1 = stats.pearsonr(x, y)[0]
        all_pearson_phi1.append(pearson_phi1)
        
        x = real_coords_spherical[:, 2]
        y = rotate_best_inf_coords_spherical[:, 2]
        pearson_phi2 = stats.pearsonr(x, y)[0]
        all_pearson_phi2.append(pearson_phi2)

    idx = np.argmax(np.abs(np.array(all_pearson_phi2)))
    theta_z = all_thetas[idx]
    print('The best Z-axis rotation angle: ', theta_z)
    rotate_best_inf_coords_euclidean = np.matmul(rotation_matrix_XY(theta_z), 
                                                best_inf_coords_euclidean.transpose()).T
    rotate_best_inf_coords_spherical = np.array(
        [euclidean_to_hyperspherical_coordinates(v) for v in np.array(rotate_best_inf_coords_euclidean)])
    
    if all_pearson_phi1[idx] < 0:
        rotate_best_inf_coords_spherical[:, 1] = np.pi - rotate_best_inf_coords_spherical[:, 1]
    if all_pearson_phi2[idx] < 0:
        rotate_best_inf_coords_spherical[:, 2] = 2*np.pi - rotate_best_inf_coords_spherical[:, 2]

    rotate_best_inf_coords_euclidean = np.array(
        [hyperspherical_to_euclidean_coordinates(v) for v in rotate_best_inf_coords_spherical])

    plt.plot(all_thetas, all_pearson_phi1, label=r'$\varphi_1$')
    plt.plot(all_thetas, all_pearson_phi2, label=r'$\varphi_2$')
    plt.ylabel('pearson correlation')
    plt.xlabel(r'$\theta_Z$')
    plt.legend()

    rotation_dict['theta_z'] = theta_z
    rotation_dict['all_pearson_phi1[idx] < 0'] = all_pearson_phi1[idx] < 0
    rotation_dict['all_pearson_phi2[idx] < 0'] = all_pearson_phi2[idx] < 0
    return rotation_dict


def apply_pipeline_matrix_maximize_pearson_get_rotated_coordinates(inf_coords_1, inf_coords_2, theta_num=50):
    rotation_dict = apply_pipeline_matrix_maximize_pearson(inf_coords_1, inf_coords_2, theta_num)
  
    inf_coords_1_values, inf_coords_2_values = normalize_coordinates(inf_coords_1, inf_coords_2)
    
    # First rotation (aligning the nodes)
    inf_coords_1_values = np.array([(rotation_dict['min_pos1_rotation_matrix'] @ v).A1 for v in inf_coords_1_values])
    inf_coords_2_values = np.array([(rotation_dict['min_pos2_rotation_matrix'] @ v).A1 for v in inf_coords_2_values])
    inf_coords_2_values = np.array(np.matmul(rotation_matrix_XY(rotation_dict['min_theta']), inf_coords_2_values.transpose()).T)

    inf_coords_spherical = np.array([euclidean_to_hyperspherical_coordinates(v) for v in inf_coords_2_values])

    # Second rotation (along X-axis)
    inf_coords_2_values = np.matmul(rotation_matrix_XY(rotation_dict['theta_z']), inf_coords_2_values.transpose()).T
    inf_coords_spherical = np.array([euclidean_to_hyperspherical_coordinates(v) for v in np.array(inf_coords_2_values)])
    if rotation_dict['all_pearson_phi1[idx] < 0']:
        inf_coords_spherical[:, 1] = np.pi - inf_coords_spherical[:, 1]
    if rotation_dict['all_pearson_phi2[idx] < 0']:
        inf_coords_spherical[:, 2] = 2*np.pi - inf_coords_spherical[:, 2]
    inf_coords_2_values = np.array([hyperspherical_to_euclidean_coordinates(v) for v in inf_coords_spherical])

    return inf_coords_1_values, inf_coords_2_values


###################### PLOTTING #############################
def plot_euclidean_coordinates_comparison(inf_coords_1_values, best_inf_coords_euclidean, title='', labels=None, 
                                          xlabel='LE+ML', ylabel='UMAP+ML'):
    plt.figure(figsize=(15, 5))
    plt.subplots_adjust(wspace=0.3)
    plt.subplot(1, 3, 1)
    y = best_inf_coords_euclidean[:, 0]
    x = inf_coords_1_values[:, 0]

    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))
    plt.xlabel(r'$X^{\textrm{' + xlabel + r'}}_1$')
    plt.ylabel(r'$X^{\textrm{' + ylabel + r'}}_1$')

    plt.subplot(1, 3, 2)
    y = best_inf_coords_euclidean[:, 1]
    x = inf_coords_1_values[:, 1]
    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))
    plt.xlabel(r'$X^{\textrm{' + xlabel + r'}}_2$')
    plt.ylabel(r'$X^{\textrm{' + ylabel + r'}}_2$')

    plt.subplot(1, 3, 3)
    y = best_inf_coords_euclidean[:, 2]
    x = inf_coords_1_values[:, 2]
    if labels is not None:
        plt.scatter(x, y, marker='.', color=[COLORS[i] for i in labels], alpha=0.8)    
    else:
        plt.scatter(x, y, marker='.', alpha=0.8)
    
    xx = np.linspace(min(np.min(x), np.min(y)), max(np.max(x), np.max(y)))
    plt.plot(xx, xx, '--', c='k')

    res = stats.pearsonr(x, y)
    plt.title('pearson={:.3f}'.format(res[0]))
    plt.xlabel(r'$X^{\textrm{' + xlabel + r'}}_3$')
    plt.ylabel(r'$X^{\textrm{' + ylabel + r'}}_3$')
    
    plt.suptitle(title, fontsize=26, y=1.02)
