import numpy as np
from . import base

from ._contact_cy import _compute_contact_matrix, _compute_edge_cache

class Manager(base.Null):
    """ Feature comparing area of contact of two segments to each segment.

    For each segment, it computes both the fraction of the segment's 'dark' 
    pixels (ie less than a specified threshold) that appear in the contact 
    area, and that value normalized by the fraction of the segment's pixels
    the contact area represents. This is motivated by inputs in which 
    pixels represent probability of a cell membrane. For those datasets,
    this feature effectively computes how much of the cell's membrane is
    touching the other cell, and how much of the contact area is membrane.
    """

    def __init__(self, thresholds=[0.1, 0.5, 0.9], oriented=False, 
                 *args, **kwargs):
        """
        Parameters
        ----------
        threshold : array-like, optional
            The 'dark' values at which the contact ratios described above
            will be computed.
        oriented : bool, optional
            Whether to use oriented probabilities.
        """
        super(Manager, self).__init__()
        self.thresholds = np.array(thresholds)
        self.oriented = oriented

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(fm_info['thresholds'], fm_info['oriented'])
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('contact')
        json_fm['contact'] = {
            'thresholds' : list(self.thresholds),
            'oriented' : self.oriented
        }
        return json_fm

    def compute_edge_features(self, g, n1, n2, cache=None):
        boundlen = len(g.boundary(n1, n2))
        volume_ratio_1 = boundlen / g.node[n1]['size']
        volume_ratio_2 = boundlen / g.node[n2]['size']
        if cache is None:
            cache = g[n1][n2][self.default_cache]
        contact_matrix = np.empty(cache.shape[:2] + (4,), dtype='double')
        _compute_contact_matrix(cache, volume_ratio_1, volume_ratio_2,
                                out=contact_matrix)
        conlen = contact_matrix.size
        feature_vector = np.zeros(conlen*2 + 4)
        feature_vector[:conlen] = contact_matrix.ravel()
        feature_vector[conlen:2*conlen] = np.log(contact_matrix.ravel())
        feature_vector[2*conlen:2*conlen+2] = np.log(np.array([volume_ratio_1, volume_ratio_2]))
        feature_vector[-1] = volume_ratio_2
        feature_vector[-2] = volume_ratio_1
        return feature_vector

    def create_edge_cache(self, g, n1, n2):
        edge_idxs = np.asarray(g.boundary(n1, n2))
        n1_idxs = np.array(list(g.extent(n1)))
        n2_idxs = np.array(list(g.extent(n2)))
        if self.oriented:
            ar = g.oriented_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r
        totals = np.empty(ar.shape[:2] + (3,), dtype='double')
        _compute_edge_cache(edge_idxs, n1_idxs, n2_idxs,
                            ar, self.thresholds, totals=totals)
        return totals

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst += src

