import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import mode
from PIL import Image
import matplotlib.pyplot as plt

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_closest_to_centroid(g):
    if len(g) == 1:
        return g[0]
    return g[np.argmin(np.linalg.norm(g[:,:3] - g[:,:3].mean(axis=0), axis=1))]


def get_closest_to_centroid_with_attributes_of_closest_to_cam(g):
    if len(g) == 1:
        return g
    xyz = g[np.argmin(np.linalg.norm(g[:,:3] - g[:,:3].mean(axis=0), axis=1)),:3]
    attributes = g[np.argmin(g[:,3], axis=0)][3:]
    return np.concatenate([xyz, attributes]).reshape(1, -1)

def remove_outliers(g):
    if len(g) == 1:
        return []
    return g

def map_3d(x, fn, size=0.03):
    # Makes Floats into Int-Bins
    to_bin = np.floor(x[:, :3] / size).astype(np.int32)
    # Lexsort Bins
    inds = np.lexsort(np.transpose(to_bin)[::-1])
    to_bin = to_bin[inds]
    x = x[inds]
    del inds
    splits = np.split(x, np.cumsum(np.unique(to_bin, return_counts=True, axis=0)[1])[:-1])
    del to_bin
    del x
    results = np.concatenate([x for x in np.vectorize(fn, otypes=[np.ndarray])(splits) if len(x)>0], axis=0)
    return results


def get_matching_indices(arr_a, arr_b):
    tree = cKDTree(arr_b)
    dist, index = tree.query(arr_a, workers=64)
    return index

def get_rotation_matrix_to_align_pose_with_gravity(pose, g):
    """Used to find rotation that rotates pose matrix to align with gravity vector g"""
    xx = np.array([0,0,1]) # Vector to which gravity is aligned
    return rotation_matrix_from_vectors(pose[:3,:3] @ (g / np.linalg.norm(g)), xx)


def get_edgeness(x):
    edgeness_x = torch.abs(x[:-1] - x[1:]) # has shape (height, width-1)
    edgeness_y = torch.abs(x[:,:-1] - x[:,1:]) # has shape (height-1, width)
    edgeness = torch.zeros_like(x)
    edgeness[:,:-1] += edgeness_y
    edgeness[:,1:] += edgeness_y
    edgeness[:-1,:] += edgeness_x
    edgeness[1:,:] += edgeness_x
    return edgeness


def aggregate_2d_grid(inp: np.ndarray, size: float) -> np.ndarray:
    """
    Builds a 2D grid along inp[:,0:2] with bin size `size`, aggregates each cell:
      - 1 point  -> keep it, count=1
      - 2 points -> keep the one with higher z (col 2), count=2
      - >=3      -> drop rows with z < mean(z), then:
                     x,y from first row in the kept set (same per bin),
                     z,r,g,b,dist,frame_idx,depth_unc are means,
                     class is the statistical mode,
                     class_rgb comes from first row with that class,
                     count is original number of points in the bin.
    Returns an array with columns:
      [bin_x, bin_y, z, r, g, b, distance_to_cam, class, class_r, class_g, class_b, frame_index, depth_unc, count]
    Then scales bin_x/bin_y by 1/size to match your original behavior.
    """
    # Guard against bad bin sizes
    if not np.isfinite(size) or size <= 0:
        span = np.ptp(inp[:, 0]) if inp.size else 1.0
        size = max(span / 1000.0, 1e-9)

    # Integer bin indices for the first two columns
    to_bin = np.floor(inp[:, 0:2] / size).astype(np.int64)

    # Sort so equal bins are contiguous
    order = np.lexsort((to_bin[:, 1], to_bin[:, 0]))
    bins_sorted = to_bin[order]
    data_sorted = inp[order]

    if bins_sorted.shape[0] == 0:
        return np.empty((0, inp.shape[1] + 1), dtype=inp.dtype)

    # Find group boundaries (run-length encoding)
    changes = np.any(np.diff(bins_sorted, axis=0) != 0, axis=1)
    boundaries = np.nonzero(np.concatenate(([True], changes, [True])))[0]

    rows = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        group = data_sorted[s:e]
        cnt = e - s
        bx, by = bins_sorted[s]  # bin indices (integers)

        if cnt == 1:
            one = group[0]
            row = np.concatenate([
                np.array([bx, by], dtype=one.dtype),
                one[2:],                      # keep original features from col 2 onward
                np.array([1], dtype=one.dtype)
            ])
            rows.append(row)
            continue

        if cnt == 2:
            chosen = group[np.argmax(group[:, 2])]
            row = np.concatenate([
                np.array([bx, by], dtype=chosen.dtype),
                chosen[2:],
                np.array([2], dtype=chosen.dtype)
            ])
            rows.append(row)
            continue

        # cnt >= 3 → filter by height >= mean
        z = group[:, 2]
        keep_mask = z >= z.mean()
        kept = group[keep_mask]
        if kept.shape[0] == 0:
            # Fallback: behave like single-point case with the first row
            one = group[0]
            row = np.concatenate([
                np.array([bx, by], dtype=one.dtype),
                one[2:],
                np.array([1], dtype=one.dtype)
            ])
            rows.append(row)
            continue

        # Unpack columns (names for clarity)
        # Expect columns: x,y,z,r,g,b,dist,cls,cr,cg,cb,frame_idx,depth_unc
        x, y, z, r, g, b, dist, cls, cr, cg, cb, frame_idx, d_unc = kept.T

        # Mode class
        mc = mode(cls, keepdims=False)[0]
        mc_mask = (cls == mc)
        mc_r = cr[mc_mask][0]
        mc_g = cg[mc_mask][0]
        mc_b = cb[mc_mask][0]

        agg = np.array([
            bx, by,              # bin indices
            z.mean(),            # mean height
            r.mean(), g.mean(), b.mean(),
            dist.mean(),         # mean distance to camera
            mc,                  # most common class
            mc_r, mc_g, mc_b,    # class color
            frame_idx.mean(),    # mean frame index
            d_unc.mean(),        # mean depth uncertainty
            cnt                  # ORIGINAL number of points in this bin
        ], dtype=kept.dtype)
        rows.append(agg)

    out = np.vstack(rows)
    
    return out

def get_legend(class_to_colors, tmp_dir):
    
    class_to_colors = dict(sorted(class_to_colors.items(), key=lambda item: len(item[0])))

    labels = list(class_to_colors.keys())
    fig,ax = plt.subplots()

    def f(m, c,l):
        return plt.plot([],[],marker=m, color=c, ls="none", label=l)[0]

    [f("s", class_color/255., class_name) for class_name, class_color in class_to_colors.items()]

    ax.axis('off')
    legend = plt.legend(ncol=5)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(tmp_dir + "/legend.png", dpi="figure", bbox_inches=bbox)
    return np.array(Image.open(tmp_dir + "/legend.png"))[:,:,:3].transpose(1, 0, 2)/255.