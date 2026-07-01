import numpy as np


class Synapse:
    def __init__(self,
                 id_: int,
                 pre_pt_root_id: int,
                 post_pt_root_id: int,
                 size: int,  # Size of the synaptic cleft in total voxel count. Generally proportional to surface area.
                 center_position: np.ndarray  # The x/y/z location of the synaptic cleft in nanometer coordinates
                 ):

        self.id_ = id_
        self.pre_pt_root_id = pre_pt_root_id
        self.post_pt_root_id = post_pt_root_id
        self.size = size
        self.center_position = center_position

        # Post-download attributes
        self.dist_to_post_syn_soma = -1.0
        self.dist_to_pre_syn_soma = -1.0

        self.depth_in_post_syn_tree = -1.0
        self.depth_in_pre_syn_tree = -1.0

        self.compartment_in_post_syn_soma = -1
        self.compartment_in_pre_syn_soma = -1

    def __repr__(self):
        return (f"Synapse(id={self.id_}, "
                f"pre_pt_root_id={self.pre_pt_root_id}, "
                f"post_pt_root_id={self.post_pt_root_id}, "
                f"size={self.size}, "
                f"center_position={self.center_position}, "
                f"dist_to_post_syn_soma={self.dist_to_post_syn_soma if hasattr(self, 'dist_to_post_syn_soma') else -1}, "
                f"dist_to_pre_syn_soma={self.dist_to_pre_syn_soma if hasattr(self, 'dist_to_pre_syn_soma') else -1}, "
                f"depth_in_post_syn_tree={self.depth_in_post_syn_tree if hasattr(self, 'depth_in_post_syn_tree') else -1}, "
                f"depth_in_pre_syn_tree={self.depth_in_pre_syn_tree if hasattr(self, 'depth_in_pre_syn_tree') else -1}, "
                )
