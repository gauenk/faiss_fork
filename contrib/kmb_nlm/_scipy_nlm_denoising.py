def _nl_means_denoising_3d(cnp.ndarray[np_floats, ndim=3] image,
                           Py_ssize_t s, Py_ssize_t d,
                           double h, double var):
    """
    Perform non-local means denoising on 3-D array
    Parameters
    ----------
    image : ndarray
        Input data to be denoised.
    s : int, optional
        Size of patches used for denoising.
    d : Py_ssize_t, optional
        Maximal distance in pixels where to search patches used for denoising.
    h : np_floats, optional
        Cut-off distance (in gray levels).
    var : np_floats
        Expected noise variance.  If non-zero, this is used to reduce the
        apparent patch distances by the expected distance due to the noise.
    Returns
    -------
    result : ndarray
        Denoised image, of same shape as input image.
    """

    if s % 2 == 0:
        s += 1  # odd value for symmetric patch

    if np_floats is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef Py_ssize_t n_pln, n_row, n_col
    n_pln, n_row, n_col = image.shape[0], image.shape[1], image.shape[2]
    cdef Py_ssize_t i_start, i_end, j_start, j_end, k_start, k_end
    cdef Py_ssize_t pln, row, col, i, j, k
    cdef Py_ssize_t offset = s / 2
    # padd the image so that boundaries are denoised as well
    cdef np_floats [:, :, ::1] padded = np.ascontiguousarray(
        np.pad(image, offset, mode='reflect'))
    cdef np_floats [:, :, ::1] result = np.empty_like(image)
    cdef np_floats new_value
    cdef np_floats weight_sum, weight

    cdef np_floats A = ((s - 1.) / 4.)
    cdef np_floats [::] range_vals = np.arange(-offset, offset + 1,
                                               dtype=dtype)
    xg_pln, xg_row, xg_col = np.meshgrid(range_vals, range_vals, range_vals,
                                         indexing='ij')
    cdef np_floats [:, :, ::1] w = np.ascontiguousarray(
        np.exp(-(xg_pln * xg_pln + xg_row * xg_row + xg_col * xg_col) /
               (2 * A * A)))
    w *= 1. / (np.sum(w) * h * h)

    cdef np_floats [:, :, :] central_patch
    var *= 2

    # Iterate over planes, taking padding into account
    with nogil:
        for pln in range(n_pln):
            i_start = pln - min(d, pln)
            i_end = pln + min(d + 1, n_pln - pln)
            # Iterate over rows, taking padding into account
            for row in range(n_row):
                j_start = row - min(d, row)
                j_end = row + min(d + 1, n_row - row)
                # Iterate over columns, taking padding into account
                for col in range(n_col):
                    k_start = col - min(d, col)
                    k_end = col + min(d + 1, n_col - col)

                    central_patch = padded[pln:pln+s, row:row+s, col:col+s]

                    new_value = 0
                    weight_sum = 0

                    # Iterate over local 3d patch for each pixel
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                weight = patch_distance_3d[np_floats](
                                    central_patch,
                                    padded[i:i+s, j:j+s, k:k+s],
                                    w, s, var)
                                # Collect results in weight sum
                                weight_sum += weight
                                new_value += weight * padded[i+offset,
                                                             j+offset,
                                                             k+offset]

                    # Normalize the result
                    result[pln, row, col] = new_value / weight_sum

    return np.asarray(result)
