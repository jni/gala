import numpy as np
from scipy import sparse

class SparseLOL:
    def __init__(self, csr):
        self.indptr = csr.indptr
        self.indices = csr.indices
        self.data = csr.data

    def __getitem__(self, item):
        if np.isscalar(item):  # get the column indices for the given row
            start, stop = self.indptr[item : item+2]
            return self.indices[start:stop]
        else:
            raise ValueError('SparseLOL can only be indexed by an integer.')


def extents(labels, input_indices=None):
    """Compute the extents of every integer value in ``arr``.

    Parameters
    ----------
    labels : array of int
        The array of values to be mapped.
    input_indices : array of int
        The indices corresponding to the label values passed. If `None`,
        we assume ``range(labels.size)``.

    Returns
    -------
    locs : sparse.csr_matrix
        A sparse matrix in which the nonzero elements of row i are the
        indices of value i in ``arr``.
    """
    if input_indices is not None:
        indices = input_indices
    else:
        indices = np.arange(labels.size, dtype=int)
    locs = sparse.csr_matrix((indices, (labels.ravel(), indices)))
    return locs
