import torch


# Fix noise strength
# Fix column index mismatch
# apply imputation fix invalid value for discrete values.


def generate_distributions_for_discrete_data(data_tensor, discrete_columns, encoder):
    probabilities = []
    for k, col_index in enumerate(discrete_columns):
        x_col = data_tensor[:, col_index]
        probabilities.append(torch.softmax(
            torch.cat([(((x_col - val) ** 2 + 1e-20) ** -1).reshape(-1, 1) for val in encoder.categories_[k]],
                      dim=-1), dim=-1))
    probabilities = torch.cat(probabilities, dim=-1)

    return probabilities


def apply_gumbel_softmax(discrete_logits, feature_sizes, temperature=0.05):
    soft_discrete_parts = []
    start = 0
    for size in feature_sizes:
        end = start + size
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(discrete_logits[:, start:end]) + 1e-10) + 1e-10)
        logits = discrete_logits[:, start:end]
        soft = torch.softmax((logits + gumbel_noise) / temperature, dim=-1)
        soft_discrete_parts.append(soft)
        start = end
    discrete_soft = torch.cat(soft_discrete_parts, dim=1)

    return discrete_soft


def estimate_CMI_soft_kronecker_gaussian(data_tensor, triplet, continuous_columns, discrete_columns, sigma=0.015,
                                         temperature=0.6):
    """
    Estimate CMI using:
    - Gaussian kernel for continuous features
    - Soft Kronecker kernel for discrete features
    """

    def gaussian_kernel(X, Y, sigma):
        diff = X[:, None, :] - Y[None, :, :]
        return torch.exp(-torch.sum(diff ** 2, dim=2) / (2 * sigma ** 2))

    def kronecker_kernel(X, Y, temperature):
        diff = torch.abs(X[:, None, :] - Y[None, :, :])
        return torch.exp(-torch.sum(diff, dim=2) / temperature)

    def kernel_matrix(data, feature_idxs):
        sub = data[:, feature_idxs]
        cont_idx = [i for i, idx in enumerate(feature_idxs) if idx in continuous_columns]
        disc_idx = [i for i, idx in enumerate(feature_idxs) if idx in discrete_columns]

        K = None
        if cont_idx:
            cont_data = sub[:, cont_idx]
            K_cont = gaussian_kernel(cont_data, cont_data, sigma)
            K = K_cont if K is None else K * K_cont
        if disc_idx:
            disc_data = sub[:, disc_idx]
            K_disc = kronecker_kernel(disc_data, disc_data, temperature)
            K = K_disc if K is None else K * K_disc

        if K is None:
            raise ValueError("No valid features provided for kernel computation.")
        return K

    def estimate_density(K):
        return torch.sum(K, dim=1) / (K.size(0) + 1e-10)

    def estimate_entropy(p):
        return -torch.mean(torch.log(torch.clamp(p, min=1e-10)))

    X, Y, Z = triplet
    K_xyz = kernel_matrix(data_tensor, [X, Y, Z])
    K_xz = kernel_matrix(data_tensor, [X, Z])
    K_yz = kernel_matrix(data_tensor, [Y, Z])
    K_z = kernel_matrix(data_tensor, [Z])

    return estimate_entropy(estimate_density(K_xz)) + \
        estimate_entropy(estimate_density(K_yz)) - \
        estimate_entropy(estimate_density(K_xyz)) - \
        estimate_entropy(estimate_density(K_z))


def estimate_CMI_gumbel_softmax_kernel(data_combined_tensor, triplet, discrete_columns, continuous_columns, encoder,
                                       sigma=0.1, temperature=0.05):
    """
    Estimate CMI using Gumbel-softmax for discrete + Gaussian kernel for combined features.
    """

    feature_sizes = [len(cats) for cats in encoder.categories_]
    total_discrete = sum(feature_sizes)

    discrete_part = data_combined_tensor[:, :total_discrete]
    discrete_logits = torch.log(discrete_part + 1e-10)
    continuous_part = data_combined_tensor[:, total_discrete:]

    discrete_soft = apply_gumbel_softmax(discrete_logits, feature_sizes, temperature=temperature)
    data_soft = torch.cat([discrete_soft, continuous_part], dim=1)

    col_map = {}
    idx = 0
    for col, size in zip(discrete_columns, feature_sizes):
        col_map[col] = (idx, idx + size)
        idx += size
    for col in continuous_columns:
        col_map[col] = (idx, idx + 1)
        idx += 1

    def compute_kernel_matrix(data, cols):
        slices = [data[:, col_map[c][0]:col_map[c][1]] for c in cols]
        selected = torch.cat(slices, dim=1)
        dist_sq = torch.cdist(selected, selected, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def estimate_density(K):
        return K.sum(dim=1) / (K.size(0) + 1e-10)

    def estimate_entropy(p):
        return -torch.mean(torch.log(torch.clamp(p, min=1e-10)))

    X, Y, Z = triplet
    K_xyz = compute_kernel_matrix(data_soft, [X, Y, Z])
    K_xz = compute_kernel_matrix(data_soft, [X, Z])
    K_yz = compute_kernel_matrix(data_soft, [Y, Z])
    K_z = compute_kernel_matrix(data_soft, [Z])

    return estimate_entropy(estimate_density(K_xz)) + \
        estimate_entropy(estimate_density(K_yz)) - \
        estimate_entropy(estimate_density(K_xyz)) - \
        estimate_entropy(estimate_density(K_z))


def estimate_CMI_separate_kernel(data_combined_tensor, triplet,
                                 discrete_columns, continuous_columns, encoder,
                                 sigma=0.1, temperature=0.05):
    """
    Estimate CMI using separate kernels for discrete (via Gumbel-softmax) and continuous (Gaussian).
    """

    feature_sizes = [len(cats) for cats in encoder.categories_]
    total_discrete = sum(feature_sizes)

    discrete_part = data_combined_tensor[:, :total_discrete]
    discrete_logits = torch.log(discrete_part + 1e-10)
    continuous_part = data_combined_tensor[:, total_discrete:]

    discrete_soft = apply_gumbel_softmax(discrete_logits, feature_sizes, temperature=temperature)

    def gaussian_kernel(X):
        dist_sq = torch.cdist(X, X, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))

    def estimate_density(K):
        return K.sum(dim=1) / (K.size(0) + 1e-10)

    def estimate_entropy(p):
        return -torch.mean(torch.log(torch.clamp(p, min=1e-10)))

    disc_idx_map = {}
    cont_idx_map = {}
    idx = 0
    for col, size in zip(discrete_columns, feature_sizes):
        disc_idx_map[col] = (idx, idx + size)
        idx += size
    for i, col in enumerate(continuous_columns):
        cont_idx_map[col] = (i, i + 1)

    def compute_kernel(d_cols, c_cols):
        n = discrete_soft.size(0)
        K_disc = torch.ones((n, n), device=discrete_soft.device)
        K_cont = torch.ones((n, n), device=discrete_soft.device)

        if d_cols:
            d_concat = torch.cat([discrete_soft[:, slice(*disc_idx_map[c])] for c in d_cols], dim=1)
            K_disc = gaussian_kernel(d_concat)

        if c_cols:
            c_concat = torch.cat([continuous_part[:, slice(*cont_idx_map[c])] for c in c_cols], dim=1)
            K_cont = gaussian_kernel(c_concat)

        return K_disc * K_cont

    def split(triplet):
        d = [i for i in triplet if i in discrete_columns]
        c = [i for i in triplet if i in continuous_columns]
        return d, c

    X, Y, Z = triplet
    d_xyz, c_xyz = split([X, Y, Z])
    d_xz, c_xz = split([X, Z])
    d_yz, c_yz = split([Y, Z])
    d_z, c_z = split([Z])

    K_xyz = compute_kernel(d_xyz, c_xyz)
    K_xz = compute_kernel(d_xz, c_xz)
    K_yz = compute_kernel(d_yz, c_yz)
    K_z = compute_kernel(d_z, c_z)

    return estimate_entropy(estimate_density(K_xz)) + \
        estimate_entropy(estimate_density(K_yz)) - \
        estimate_entropy(estimate_density(K_xyz)) - \
        estimate_entropy(estimate_density(K_z))


def compute_all_cmi_methods(data_tensor, data_combined_tensor,
                            triplet, encoder,
                            discrete_columns, continuous_columns,
                            sigma=0.1, temperature=0.05):
    """
    Compute three CMI values using different kernel estimation strategies, all with torch support.

    :param data_tensor: torch.Tensor for mixed kernel method (original data)
    :param data_combined_tensor: torch.Tensor for Method 2 (Gumbel + combined features)
    :param triplet: tuple of (X, Y, Z) column names
    :param encoder: fitted OneHotEncoder
    :param discrete_columns: list of discrete column names
    :param continuous_columns: list of continuous column names
    :param sigma: float, kernel bandwidth
    :param temperature: float, Gumbel-softmax temperature
    :return: tuple of (cmi1, cmi2, cmi3)
    """
    cmi1 = estimate_CMI_soft_kronecker_gaussian(
        data_tensor, triplet, continuous_columns, discrete_columns,
        sigma=sigma, temperature=temperature
    )

    cmi2 = estimate_CMI_gumbel_softmax_kernel(
        data_combined_tensor, triplet, discrete_columns, continuous_columns,
        encoder=encoder, sigma=sigma, temperature=temperature
    )

    cmi3 = estimate_CMI_separate_kernel(
        data_combined_tensor, triplet,
        discrete_columns, continuous_columns, encoder=encoder,
        sigma=sigma, temperature=temperature
    )

    return cmi1, cmi2, cmi3