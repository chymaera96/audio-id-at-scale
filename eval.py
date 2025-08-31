# eval.py — simplified for top-1 exact @ 1-fingerprint queries
# This code is sourced from https://github.com/mimbres/neural-audio-fp.git

import faiss
import time
import numpy as np
import os
import uuid

def get_index(index_type,
              train_data,
              train_data_shape,
              use_gpu=True,
              max_nitem_train=2e7,
              n_centroids=64,
):
    """
    • Create FAISS index
    • Train index using (partial) data
    • Return index
    Parameters
    ----------
    index_type : (str)
        Index type must be one of {'L2', 'IVF', 'IVFPQ', 'IVFPQ-RR',
                                   'IVFPQ-ONDISK', HNSW'}
    train_data : (float32)
        numpy.memmap or numpy.ndarray
    train_data_shape : list(int, int)
        Data shape (n, d). n is the number of items. d is dimension.
    use_gpu: (bool)
        If False, use CPU. Default is True.
    max_nitem_train : (int)
        Max number of items to be used for training index. Default is 1e7.
    Returns
    -------
    index : (faiss.swigfaiss_avx2.GpuIndex***)
        Trained FAISS index.
    References:
        https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """
    # GPU Setup
    if use_gpu:
        GPU_RESOURCES = faiss.StandardGpuResources()
        GPU_OPTIONS = faiss.GpuClonerOptions()
        GPU_OPTIONS.useFloat16 = True
    else:
        pass

    # Fingerprint dimension, d
    d = train_data_shape[1]

    # Build a flat (CPU) index
    index = faiss.IndexFlatL2(d)

    mode = index_type.lower()
    print(f'Creating index: \x1b[93m{mode}\x1b[0m')
    if mode == 'l2':
        pass
    elif mode == 'ivf':
        nlist = 400
        index = faiss.IndexIVFFlat(index, d, nlist)
    elif mode == 'ivfpq':
        code_sz = 64
        nbits = 8
        index = faiss.IndexIVFPQ(index, d, n_centroids, code_sz, nbits)
    elif mode == 'lsh':
        nbits = 256
        index = faiss.IndexLSH(d, nbits)
    elif mode == 'ivfpq-rr':
        code_sz = 64
        nbits = 8
        M_refine = 4
        nbits_refine = 4
        index = faiss.IndexIVFPQR(index, d, n_centroids, code_sz, nbits,
                                  M_refine, nbits_refine)
    elif mode == 'ivfpq-ondisk':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        raise NotImplementedError(mode)
    elif mode == 'hnsw':
        if use_gpu:
            raise NotImplementedError(f'{mode} is only available in CPU.')
        else:
            M = 16
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efConstruction = 80
            index.verbose = True
            index.hnsw.search_bounded_queue = True
    else:
        raise ValueError(mode.lower())

    # From CPU index to GPU index
    if use_gpu:
        print('Copy index to \x1b[93mGPU\x1b[0m.')
        index = faiss.index_cpu_to_gpu(GPU_RESOURCES, 0, index, GPU_OPTIONS)

    # Train index
    start_time = time.time()
    if len(train_data) > max_nitem_train:
        print('Training index using {:>3.2f} % of data...'.format(
            100. * max_nitem_train / len(train_data)))
        sel_tr_idx = np.random.permutation(len(train_data))
        sel_tr_idx = sel_tr_idx[:int(max_nitem_train)]
        index.train(train_data[sel_tr_idx,:])
    else:
        print('Training index...')
        index.train(train_data)  # No-op for {'l2', 'hnsw'}
    print('Elapsed time: {:.2f} seconds.'.format(time.time() - start_time))

    index.nprobe = 20
    return index


def load_memmap_data(source_dir,
                     fname,
                     append_extra_length=None,
                     shape_only=False,
                     display=True):
    """
    Load data and datashape from the file path.
    • Get shape from [source_dir/fname_shape.npy}.
    • Load memmap data from [source_dir/fname.mm].
    """
    path_shape = os.path.join(source_dir, fname + '_shape.npy')
    path_data = os.path.join(source_dir, fname + '.mm')
    data_shape = np.load(path_shape)
    if shape_only:
        return data_shape

    if append_extra_length:
        data_shape[0] += append_extra_length
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    else:
        data = np.memmap(path_data, dtype='float32', mode='r+',
                         shape=(data_shape[0], data_shape[1]))
    # Convert nan values to 0
    data[np.isnan(data)] = 0.0
    if display:
        print(f'Load {data_shape[0]:,} items from \x1b[32m{path_data}\x1b[0m.')
    return data, data_shape


def eval_faiss(emb_dir,
               emb_dummy_dir=None,
               num_dummy=None,
               index_type='ivfpq',
               nogpu=False,
               max_train=1e7,
               test_ids='icassp',
               test_seq_len='1',
               k_probe=20,
               n_centroids=64):
    """
    Segment/sequence-wise audio search experiment and evaluation: FAISS-based.
    Simplified for this project to compute ONLY top-1 exact hit rate at a
    single-segment (1 fingerprint) query length.
    """
    # Ensure sequence length is exactly [1]
    if isinstance(test_seq_len, str):
        test_seq_len = list(map(int, test_seq_len.split()))
    test_seq_len = [1]

    # Load items from {query, db, dummy_db}
    query, query_shape = load_memmap_data(emb_dir, 'query')
    db, db_shape = load_memmap_data(emb_dir, 'db')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    else:
        print(f'Using \x1b[93m{emb_dummy_dir}\x1b[0m as dummy embedding directory...')
    dummy_db, dummy_db_shape = load_memmap_data(emb_dummy_dir, 'dummy_db')

    if num_dummy is not None and num_dummy < dummy_db_shape[0]:
        dummy_db = dummy_db[:num_dummy, :]
        dummy_db_shape[0] = num_dummy
        print(f'Using only {num_dummy} items from dummy_db.')

    # Create and train FAISS index
    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                      max_train, n_centroids=n_centroids)

    # Add items to index
    start_time = time.time()
    index.add(dummy_db); print(f'{len(dummy_db)} items from dummy DB')
    index.add(db); print(f'{len(db)} items from reference DB')
    t = time.time() - start_time
    print(f'Added total {index.ntotal} items to DB. {t:>4.2f} sec.')

    # Prepare fake reconstruction index (dummy + db)
    del dummy_db
    start_time = time.time()
    fake_recon_index, index_shape = load_memmap_data(
        emb_dummy_dir, 'dummy_db', append_extra_length=query_shape[0],
        display=False)
    fake_recon_index[dummy_db_shape[0]:dummy_db_shape[0] + query_shape[0], :] = db[:, :]
    fake_recon_index.flush()
    print(f'Created fake_recon_index, total {index_shape[0]} items. {time.time() - start_time:>4.2f} sec.')

    # Get test_ids
    print(f'test_id: \x1b[93m{test_ids}\x1b[0m,  ', end='')
    if isinstance(test_ids, str) and test_ids.lower() == 'all':
        test_ids = np.arange(0, len(query) - 1, 1)
    elif isinstance(test_ids, str) and test_ids.isnumeric():
        np.random.seed(42)
        test_ids = np.random.permutation(len(query) - 1)[:int(test_ids)]
    elif isinstance(test_ids, str):
        test_ids = np.load(test_ids)
    else:
        test_ids = np.asarray(test_ids)

    n_test = len(test_ids)
    gt_ids  = test_ids + dummy_db_shape[0]
    print(f'n_test: \x1b[93m{n_test:n}\x1b[0m')

    # Metric buffers (n_test, 1)
    top1_exact = np.zeros((n_test, 1), dtype=int)

    # Evaluation (sl is always 1)
    for ti, test_id in enumerate(test_ids):
        gt_id = gt_ids[ti]
        q = query[test_id:(test_id + 1), :]  # shape (1, d)

        # segment-level top-k search
        _, I = index.search(q, k_probe)
        # offset compensation (no-op for sl=1 but keep for consistency)
        I[0, :] -= 0
        candidates = np.unique(I[I >= 0])

        # sequence score degenerates to single dot for sl=1
        _scores = np.zeros(len(candidates))
        for ci, cid in enumerate(candidates):
            _scores[ci] = float(np.dot(q[0], fake_recon_index[cid:cid + 1, :].T))

        pred_ids = candidates[np.argsort(-_scores)[:1]]
        top1_exact[ti, 0] = int(gt_id == pred_ids[0])

    # Summary
    top1_exact_rate = float(100. * np.mean(top1_exact, axis=0)[0])

    # Persist results
    result_dir = emb_dir + f'/{str(uuid.uuid4().hex)[:8]}'
    try:
        os.makedirs(result_dir, exist_ok=True)
    except OSError:
        print(f'Failed to create directory {result_dir}.')
        result_dir = emb_dir + f'/{str(uuid.uuid4().hex)[:16]}'

    np.save(f'{result_dir}/top1_exact_rate.npy', np.array([top1_exact_rate], dtype=np.float32))
    np.save(f'{result_dir}/raw_top1.npy', top1_exact)
    np.save(f'{emb_dir}/test_ids.npy', test_ids)
    print(f'Saved test_ids, top1_exact_rate and raw_top1 to {result_dir}.')

    return top1_exact_rate


