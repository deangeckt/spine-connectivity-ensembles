import pickle
from scipy.sparse import spmatrix
from scipy.sparse import load_npz, save_npz


def save_connectivity(sparse_mat: spmatrix, mapping: dict, path: str, name: str):
    save_npz(f'{path}/{name}_matrix.npz', sparse_mat)
    with open(f'{path}/{name}_mapping.pkl', 'wb') as fp:
        pickle.dump(mapping, fp)


def load_connectivity(path: str, name: str) -> tuple[spmatrix, dict]:
    sparse_matrix = load_npz(f'{path}/{name}_matrix.npz')
    with open(f'{path}/{name}_mapping.pkl', 'rb') as fp:
        mapping = pickle.load(fp)
    return sparse_matrix, mapping
