import cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_contact_matrix(double[:, :, :] totals, double volume_ratio_1,
                            double volume_ratio_2, double[:, :, :] out):
    cdef int tt, cc, feature_count, nchannels, nthresholds
    nchannels = totals.shape[0]
    nthresholds = totals.shape[1]
    feature_count = 4
    for cc in range(nchannels): # for each channel
        for tt in range(nthresholds): # for each threshold
            # for n1, ratio of dark pixels in boundary to rest of segment 1
            out[cc, tt, 0] = totals[cc, tt, 0] / totals[cc, tt, 1]
            # same normalized by ratio of total pixels
            out[cc, tt, 1] = out[cc, tt, 0] / volume_ratio_1
            # same for segment 2
            out[cc, tt, 2] = totals[cc, tt, 0] / totals[cc, tt, 2]
            out[cc, tt, 3] = out[cc, tt, 2] / volume_ratio_2


@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_edge_cache(Py_ssize_t[:] edge_idxs, Py_ssize_t[:] n1_idxs,
                        Py_ssize_t[:] n2_idxs,
                        double[:,:] vals, double[:] thresholds,
                        double[:, :, :] totals):

    cdef int tt, cc, nchannels, nthresholds
    nchannels = vals.shape[1]
    nthresholds = thresholds.shape[0]
    for cc in range(nchannels): # for each channel
        for tt in range(nthresholds): # for each threshold
            for nn in range(edge_idxs.shape[0]): # for each voxel
                if vals[edge_idxs[nn],cc] > thresholds[tt]: totals[cc, tt, 0] += 1
            for nn in range(n1_idxs.shape[0]): # for each voxel
                if vals[n1_idxs[nn],cc] > thresholds[tt]: totals[cc, tt, 1] += 1
            for nn in range(n2_idxs.shape[0]): # for each voxel
                if vals[n2_idxs[nn],cc] > thresholds[tt]: totals[cc, tt, 2] += 1
