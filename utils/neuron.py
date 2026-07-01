import os
import pickle

import numpy as np
from typing import Optional

from synapse import Synapse
from connectome_types import ClfType, SKELETONS_DIR_PATH

from meshparty.skeleton import Skeleton
from skeleton_plot import skel_io
import pandas as pd


def read_local_swc(file_path):
    def apply_casts(df, casts):
        for key, typ in casts.items():
            df[key] = df[key].astype(typ)

    df = pd.read_csv(file_path, names=skel_io.SWC_COLUMNS, comment='#', sep=' ')
    apply_casts(df, skel_io.COLUMN_CASTS)

    if not all(df.index == df['id']):
        id_map = dict(zip(df['id'], df.index))
        id_map[-1] = -1
        df['id'] = [id_map[x] for x in df['id']]
        df['parent'] = [id_map[x] for x in df['parent']]

    verts = df[['x', 'y', 'z']].values
    edges = df[['id', 'parent']].iloc[1:].values
    sk = Skeleton(verts, edges, vertex_properties={'radius': df['radius'],
                                                   'compartment': df['type']}, root=0,
                  remove_zero_length_edges=False)
    return sk


class Neuron:
    def __init__(self,
                 root_id: int,
                 clf_type: ClfType,
                 cell_type: str,
                 mtype: int,
                 position: np.ndarray,
                 volume: float,
                 pre_synapses: list[Synapse],
                 post_synapses: list[Synapse]):

        self.nucleus_id = None  # consistent, exist in EM_NEURONS_PATH neurons
        self.root_id = root_id
        self.clf_type = clf_type
        self.cell_type = cell_type
        self.mtype = mtype
        self.position = position
        self.volume = volume

        self.post_synapses = post_synapses  # outgoing
        self.pre_synapses = pre_synapses  # incoming

        # ds refer to the whole dataset, not just within the column
        self.ds_num_of_incoming_synapses = -1
        self.ds_num_of_outgoing_synapses = -1

        self.ds_incoming_syn_mean_weight = -1
        self.ds_outgoing_syn_mean_weight = -1

        self.ds_incoming_syn_std_weight = -1
        self.ds_outgoing_syn_std_weight = -1

        self.ds_incoming_syn_sum_weight = -1
        self.ds_outgoing_syn_sum_weight = -1

        self.axon_length = -1
        self.dendrite_length = -1
        self.max_dend_y_path_dist_to_soma = -1
        self.max_dend_y_euclid_dist_to_soma = -1

    def __repr__(self):
        return (f"Neuron(root_id={self.root_id},"
                f"clf_type={self.clf_type.value},"
                f"cell_type={self.cell_type},"
                f"mtype={self.mtype},"
                f"position={self.position},"
                f"volume={self.volume},"
                )

    def load_skeleton(self, s3=False) -> Optional[Skeleton]:
        return self.__load_skeleton_swc()

    def __load_skeleton_swc(self) -> Optional[Skeleton]:
        cell_id = self.root_id
        sk_file_path = f'{SKELETONS_DIR_PATH}/{cell_id}.swc'
        if not os.path.exists(sk_file_path) or os.path.getsize(sk_file_path) == 0:
            return None
        return read_local_swc(sk_file_path)


