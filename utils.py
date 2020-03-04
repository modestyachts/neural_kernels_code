import concurrent.futures as fs
import  os
import pickle
import io
import boto3 
from numba import jit
import numpy as np
import scipy.misc
import s3_utils
import math
from config import CN
from skimage import feature
from PIL import Image

def pretty_time_delta(seconds):
    sign_string = '-' if seconds < 0 else ''
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%s%dd%dh%dm%ds' % (sign_string, days, hours, minutes, seconds)
    elif hours > 0:
        return '%s%dh%dm%ds' % (sign_string, hours, minutes, seconds)
    elif minutes > 0:
        return '%s%dm%ds' % (sign_string, minutes, seconds)
    else:
        return '%s%ds' % (sign_string, seconds)

def softmax(predictions):
    """
    Args:
        predictions (num_data, num_labels)
    """
    predictions = np.exp(predictions - predictions.max(axis=1, keepdims=True))
    return predictions / predictions.sum(axis=1, keepdims=True)

@jit(nopython=True)
def fast_exp_ip(K, gamma):
    for x in range(K.shape[0]):
        for y in range(K.shape[1]):
            K[x,y] = math.exp(gamma*K[x,y])
    return K

def torch_eval(net, loader):
    logits = []
    truth = []
    if (torch.cuda.is_available()):
        net = net.cuda()
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if (torch.cuda.is_available()):
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        logits.append(outputs.cpu().detach().numpy())
        truth.append(labels.cpu().detach().numpy())
    logits = np.vstack(logits)
    truth = np.hstack(truth)
    return logits, truth

def make_torch_dataset_from_numpy(X, y, shuffle=False, dtype=np.float32, num_channels=3, input_ord="HWC", batch_size=128):
    if (input_ord == "HWC"):
        X = X.transpose(0, 3, 1, 2)
    elif (input_ord == "CHW"):
        pass
    else:
        raise Exception("unsupported data order")
    print(X.shape)
    assert X.shape[1] == num_channels
    X = X.astype(dtype)
    X_torch = torch.from_numpy(X)
    y_torch = torch.from_numpy(y)
    dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
    return dataset

def make_torch_loader_from_numpy(X, y, shuffle=False, dtype=np.float32, num_channels=3, input_ord="HWC", batch_size=128):

    dataset = make_torch_dataset_from_numpy(X, y, shuffle, dtype, num_channels, input_ord, batch_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return loader

def key_exists(bucket, key):
    # Return true if a key exists in s3 bucket
    client = get_s3_client()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as exc:
        if exc.response['Error']['Code'] != '404':
            raise
        return False
    except:
        raise


def serialize_torch_network(net):
    bio = io.BytesIO()
    torch.save(net, bio)
    return bio.getvalue()

def numpy_to_bytes(ar):
    bio = io.BytesIO()
    np.save(bio, ar)
    return bio.getvalue()

def bytes_to_numpy(barray):
    bio = io.BytesIO(barray)
    return np.load(bio)

def deserialize_torch_network(net_bytes):
    return torch.load(io.BytesIO(net_bytes))

def load_dataset(dataset_name, bucket="pictureweb", cache=True):
    # TODO: move this to utils and make it use S3
    wrapper = s3_utils.S3Wrapper(bucket=bucket, cache_on_local_disk=cache)
    if dataset_name == 'cifar-10':
        return np.load(io.BytesIO(wrapper.get(key="datasets/cifar_10_zca_augmented_extra_zca_augment_en9fKkGMMg.npz")),  allow_pickle=True)
    elif dataset_name == "cifar-100":
        return np.load(io.BytesIO(wrapper.get(key="datasets/cifar_100_zca_augmented_u3jS2A3Qww.npz")), allow_pickle=True)
    elif dataset_name == 'mnist':
        return np.load(io.BytesIO(wrapper.get(key="datasets/mnist.npz")), allow_pickle=True)
    elif dataset_name == 'cifar-10-raw':
        return np.load(io.BytesIO(wrapper.get(key="datasets/cifar.npz")),  allow_pickle=True)
    else:
        raise Exception('Unknown dataset "{}"'.format(dataset_name))

def rbf_kernel_numpy(X, Y, gamma):
    K = X.dot(Y.T)
    K *= -2
    K += (np.linalg.norm(X, axis=1)**2)[:, np.newaxis]
    K += (np.linalg.norm(Y, axis=1)**2)[:, np.newaxis].T
    K *= -1*gamma
    return fast_exp_ip(K, gamma)


@jit(nogil=True, cache=True)
def __grab_patches(images, random_idxs, patch_size=6, tot_patches=1e6, seed=0, scale=0):
    patches = np.zeros((len(random_idxs), images.shape[1], patch_size, patch_size), dtype=images.dtype)
    for i, (im_idx, idx_x, idx_y) in enumerate(random_idxs):
        out_patch = patches[i, :, :, :]
        im = images[im_idx]
        if (scale != 0):
            im = skimage.filters.gaussian(im, sigma=scale)
        grab_patch_from_idx(im, idx_x, idx_y, patch_size, out_patch)
    return patches


@jit(nopython=True, nogil=True)
def grab_patch_from_idx(im, idx_x, idx_y, patch_size, outpatch):
    sidx_x = int(idx_x - patch_size/2)
    eidx_x = int(idx_x + patch_size/2)
    sidx_y = int(idx_y - patch_size/2)
    eidx_y = int(idx_y + patch_size/2)
    outpatch[:,:,:] = im[:, sidx_x:eidx_x, sidx_y:eidx_y]
    return outpatch

def grab_patches(images, patch_size=6, tot_patches=5e5, seed=0, max_threads=50, scale=0, rgb=True):
    if (rgb):
        images = images.transpose(0, 3, 1, 2)
    idxs = chunk_idxs(images.shape[0], max_threads)
    tot_patches = int(tot_patches)
    patches_per_thread = int(tot_patches/max_threads)
    np.random.seed(seed)
    seeds = np.random.choice(int(1e5), len(idxs), replace=False)
    dtype = images.dtype

    tot_patches = int(tot_patches)



    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i,(sidx, eidx) in enumerate(idxs):
            images.shape[0]
            im_idxs = np.random.choice(images[sidx:eidx, :].shape[0], patches_per_thread)
            idxs_x = np.random.choice(int(images.shape[2]) - patch_size - 1, tot_patches)
            idxs_y = np.random.choice(int(images.shape[3]) - patch_size - 1, tot_patches)
            idxs_x += int(np.ceil(patch_size/2))
            idxs_y += int(np.ceil(patch_size/2))
            random_idxs =  list(zip(im_idxs, idxs_x, idxs_y))

            futures.append(executor.submit(__grab_patches, images[sidx:eidx, :],
                                           patch_size=patch_size,
                                           random_idxs=random_idxs,
                                           tot_patches=patches_per_thread,
                                           seed=seeds[i],
                                           scale=scale
                                            ))
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    idxs = np.random.choice(results.shape[0], results.shape[0], replace=False)
    return results[idxs], idxs

def patchify_all_imgs(X, patch_shape):
    out = []
    i = 0
    for x in X:
        dim = x.shape[0]
        patches = patchify(x, patch_shape)
        out_shape = patches.shape
        out.append(patches.reshape(out_shape[0]*out_shape[1], patch_shape[0], patch_shape[1], -1))
    return np.array(out)

def patchify(img, patch_shape):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always
    img = np.ascontiguousarray(img)  # won't make a copy if not needed
    X, Y, Z = img.shape
    x, y= patch_shape
    shape = ((X-x+1), (Y-y+1), x, y, Z) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
#    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y*Z, Z, Y*Z, Z, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches



def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar(path=".", center=False):
    train_batches = []
    train_labels = []
    for i in range(1,6):
        cifar_out = unpickle(os.path.join(path, "data_batch_{0}".format(i)))
        train_batches.append(cifar_out[b"data"])
        train_labels.extend(cifar_out[b"labels"])
    # Stupid bull shit to get pixels in correct order
    X_train= np.vstack(tuple(train_batches)).reshape(-1, 32*32, 3)
    X_train = X_train.reshape(-1,3,32,32).transpose(0,2,3,1).reshape(-1,32,32, 3)
    y_train = np.array(train_labels)
    cifar_out = unpickle(os.path.join(path, "test_batch"))
    X_test = cifar_out[b"data"].reshape(-1, 32*32, 3)
    X_test = X_test.reshape(-1,3,32,32).transpose(0,2,3,1).reshape(-1,32,32, 3)
    y_test = cifar_out[b"labels"]

    return (X_train, np.array(y_train)), (X_test, np.array(y_test))

def normalize_patches(patches, min_divisor=1e-8, zca_bias=0.001, mean_rgb=np.array([0,0,0])):
    if (patches.dtype == 'uint8'):
        patches = patches.astype('float64')
        patches /= 255.0
    print("zca bias", zca_bias)
    n_patches = patches.shape[0]
    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)
    # Zero mean every feature
    patches = patches - np.mean(patches, axis=1)[:,np.newaxis]

    # Normalize
    patch_norms = np.linalg.norm(patches, axis=1)

    # Get rid of really small norms
    patch_norms[np.where(patch_norms < min_divisor)] = 1

    # Make features unit norm
    patches = patches/patch_norms[:,np.newaxis]


    patchesCovMat = 1.0/n_patches * patches.T.dot(patches)

    (E,V) = np.linalg.eig(patchesCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    patches_normalized = (patches).dot(global_ZCA).dot(global_ZCA.T)

    return patches_normalized.reshape(orig_shape).astype('float32')

def preprocess(train, test, min_divisor=1e-8, zca_bias=0.0001, return_weights=False):
    origTrainShape = train.shape
    origTestShape = test.shape



    train = np.ascontiguousarray(train, dtype=np.float32).reshape(train.shape[0], -1).astype('float64')
    test = np.ascontiguousarray(test, dtype=np.float32).reshape(test.shape[0], -1).astype('float64')


    nTrain = train.shape[0]

    # Zero mean every feature
    train = train - np.mean(train, axis=1)[:,np.newaxis]
    test = test - np.mean(test, axis=1)[:,np.newaxis]

    # Normalize
    train_norms = np.linalg.norm(train, axis=1)
    test_norms = np.linalg.norm(test, axis=1)

    # Make features unit norm
    train = train/train_norms[:,np.newaxis]
    test = test/test_norms[:,np.newaxis]

    data_means = np.mean(train, axis=1)


    trainCovMat = 1.0/nTrain * train.T.dot(train)

    (E,V) = np.linalg.eig(trainCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)

    train = (train).dot(global_ZCA)
    test = (test).dot(global_ZCA)
    if return_weights:
        return (train.reshape(origTrainShape).astype('float64'), test.reshape(origTestShape).astype('float64')), global_ZCA
    else:
        return (train.reshape(origTrainShape).astype('float64'), test.reshape(origTestShape).astype('float64'))

def chunk_idxs(size, chunks):
    chunk_size  = int(np.ceil(size/chunks))
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))

def chunk_idxs_by_size(size, chunk_size):
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def top_k_accuracy(labels, y_pred, k=5):
    top_k_preds = get_top_k(y_pred, k=k)
    if (len(labels.shape) == 1):
        labels = labels[:, np.newaxis]
    correct = np.sum(np.any(top_k_preds == labels, axis=1))
    return correct/float(labels.shape[0])

def get_top_k(y_pred, k=5, threads=70):
    with fs.ThreadPoolExecutor(max_workers=threads) as executor:
        idxs = chunk_idxs(y_pred.shape[0], threads)
        futures = []
        for (sidx, eidx) in idxs:
            futures.append(executor.submit(_get_top_k, y_pred[sidx:eidx, :], k))
        fs.wait(futures)
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    return results

def list_all_keys(prefix, bucket='robustcifar'):
    client = get_s3_client()
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix)
    if (objects.get('Contents') == None):
        return []
    keys = list(map(lambda x: x['Key'], objects.get('Contents', [] )))
    truncated = objects['IsTruncated']
    next_marker = objects.get('NextMarker')
    while truncated:
        objects = client.list_objects(Bucket=bucket, Prefix=prefix,
                                      Delimiter=prefix, Marker=next_marker)
        truncated = objects['IsTruncated']
        next_marker = objects.get('NextMarker')
        keys += list(map(lambda x: x['Key'], objects['Contents']))
    return list(filter(lambda x: len(x) > 0, keys))

@jit(nopython=True, nogil=True)
def _get_top_k(y_pred, k=5):
    top_k_preds = np.ones((y_pred.shape[0], k))
    top_k_pred_weights = np.ones((y_pred.shape[0], k))

    top_k_pred_weights *= -99999999
    for i in range(y_pred.shape[0]):
        top_k = top_k_preds[i, :]
        top_k_pred_weights_curr = top_k_pred_weights[i, :]
        for j in range(y_pred.shape[1]):
            in_top_k = False
            for elem in top_k_pred_weights_curr:
                in_top_k = in_top_k | (y_pred[i,j] > elem)
            if (in_top_k):
                min_idx = 0
                for z in range(top_k_pred_weights_curr.shape[0]):
                    if top_k_pred_weights_curr[min_idx] > top_k_pred_weights_curr[z]:
                        min_idx = z
                top_k[min_idx] = j
                top_k_pred_weights_curr[min_idx] = y_pred[i,j]
    return top_k_preds

def apply_zca(X, zca):
    old_shape = X.shape
    X = X.reshape(old_shape[0], -1)
    X = X - np.mean(X, axis=1)[:, np.newaxis]
    norms = np.linalg.norm(X, axis=1)
    X = X / norms[:, np.newaxis]
    X = X.dot(zca).reshape(old_shape)
    return X

def apply_transform(f, X):
    X_out = X.copy()
    assert X.dtype == np.uint8
    for i,x in enumerate(X):
        X_out[i] = np.array(f(Image.fromarray(x)))[:, :, :3]
    return X_out


def cfg_to_dict(cfg):
    output = {}
    for k,v in cfg.items():
        if isinstance(v, CN):
            output[k] = cfg_to_dict(v)
        else:
            output[k] = v
    return output

def corner(x):
    corners = []
    for i in range(3):
        corners.append(feature.corner_harris(x[:, :, i])[:, :, np.newaxis])
    return np.concatenate([x] + corners, axis=2)

def corners_full(X):
    X_corners = []
    for x in X:
        X_corners.append(corner(x))
    return np.stack(X_corners, axis=0)
