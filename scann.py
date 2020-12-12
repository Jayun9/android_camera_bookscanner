import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

PAGE_MARGIN_X = 20
PAGE_MARGIN_Y = 20
OUTPUT_ZOOM = 1.0
OUTPUT_DPI = 300
REMAP_DECIMATE = 16
ADAPTIVE_WINSZ = 55
TEXT_MIN_WIDTH = 15
TEXT_MIN_HEIGHT = 2
TEXT_MIN_ASPECT = 1.5
TEXT_MAX_THICKNESS = 10
EDGE_MAX_OVERLAP = 1.0
EDGE_MAX_LENGTH = 100.0
EDGE_ANGLE_COST = 10.0
EDGE_MAX_ANGLE = 7.5
RVEC_IDX = slice(0, 3)
TVEC_IDX = slice(3, 6)
CUBIC_IDX = slice(6, 8)
SPAN_MIN_WIDTH = 30
SPAN_PX_PER_STEP = 20
FOCAL_LENGTH = 1.2
K = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]], dtype=np.float32)

def round_nearest_multiple(i, factor):
    i = int(i)
    rem = i % factor
    if not rem:
        return i
    else:
        return i + factor - rem

def pix2norm(shape, pts):
    height, width = shape[:2]
    scl = 2.0/(max(height, width))
    offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
    return (pts - offset) * scl

def norm2pix(shape, pts, as_integer):
    height, width = shape[:2]
    scl = max(height, width)*0.5
    offset = np.array([0.5*width, 0.5*height],
                      dtype=pts.dtype).reshape((-1, 1, 2))
    rval = pts * scl + offset
    if as_integer:
        return (rval + 0.5).astype(int)
    else:
        return rval

def get_default_params(corners, ycoords, xcoords):
    page_width = np.linalg.norm(corners[1] - corners[0])
    page_height = np.linalg.norm(corners[-1] - corners[0])
    rough_dims = (page_width, page_height)
    cubic_slopes = [0.0, 0.0]
    corners_object3d = np.array([
        [0, 0, 0],
        [page_width, 0, 0],
        [page_width, page_height, 0],
        [0, page_height, 0]])
    _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                 corners, K, np.zeros(5))
    span_counts = [len(xc) for xc in xcoords]
    params = np.hstack((np.array(rvec).flatten(),
                        np.array(tvec).flatten(),
                        np.array(cubic_slopes).flatten(),
                        ycoords.flatten()) +
                       tuple(xcoords))
    return rough_dims, span_counts, params

def project_xy(xy_coords, pvec):
    alpha, beta = tuple(pvec[CUBIC_IDX])
    poly = np.array([
        alpha + beta,
        -2*alpha - beta,
        alpha,
        0])
    xy_coords = xy_coords.reshape((-1, 2))
    z_coords = np.polyval(poly, xy_coords[:, 0])
    objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))
    image_points, _ = cv2.projectPoints(objpoints,
                                        pvec[RVEC_IDX],
                                        pvec[TVEC_IDX],
                                        K, np.zeros(5))
    return image_points

def project_keypoints(pvec, keypoint_index):
    xy_coords = pvec[keypoint_index]
    xy_coords[0, :] = 0
    return project_xy(xy_coords, pvec)

def resize_to_screen(src, maxw=1280, maxh=700):
    height, width = src.shape[:2]
    scl_x = float(width)/maxw
    scl_y = float(height)/maxh
    scl = int(np.ceil(max(scl_x, scl_y)))
    if scl > 1.0:
        inv_scl = 1.0/scl
        img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
    else:
        img = src
    return img

def box(width, height):
    return np.ones((height, width), dtype=np.uint8)

def get_page_extents(small):
    height, width = small.shape[:2]
    xmin = PAGE_MARGIN_X
    ymin = PAGE_MARGIN_Y
    xmax = width-PAGE_MARGIN_X
    ymax = height-PAGE_MARGIN_Y
    page = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)
    outline = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]])
    return page, outline

def get_mask(small, pagemask):
    sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV,
                                 ADAPTIVE_WINSZ,
                                 25)
    mask = cv2.dilate(mask, box(9, 1))
    mask = cv2.erode(mask, box(1, 3))
    return np.minimum(mask, pagemask)

def interval_measure_overlap(int_a, int_b):
    return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])

def angle_dist(angle_b, angle_a):
    diff = angle_b - angle_a
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return np.abs(diff)

def blob_mean_and_tangent(contour):
    moments = cv2.moments(contour)
    area = moments['m00']
    mean_x = moments['m10'] / area
    mean_y = moments['m01'] / area
    moments_matrix = np.array([
        [moments['mu20'], moments['mu11']],
        [moments['mu11'], moments['mu02']]
    ]) / area
    _, svd_u, _ = cv2.SVDecomp(moments_matrix)
    center = np.array([mean_x, mean_y])
    tangent = svd_u[:, 0].flatten().copy()
    return center, tangent

class ContourInfo(object):

    def __init__(self, contour, rect, mask):
        self.contour = contour
        self.rect = rect
        self.mask = mask
        self.center, self.tangent = blob_mean_and_tangent(contour)
        self.angle = np.arctan2(self.tangent[1], self.tangent[0])
        clx = [self.proj_x(point) for point in contour]
        lxmin = min(clx)
        lxmax = max(clx)
        self.local_xrng = (lxmin, lxmax)
        self.point0 = self.center + self.tangent * lxmin
        self.point1 = self.center + self.tangent * lxmax
        self.pred = None
        self.succ = None

    def proj_x(self, point):
        return np.dot(self.tangent, point.flatten()-self.center)

    def local_overlap(self, other):
        xmin = self.proj_x(other.point0)
        xmax = self.proj_x(other.point1)
        return interval_measure_overlap(self.local_xrng, (xmin, xmax))

def generate_candidate_edge(cinfo_a, cinfo_b):
    if cinfo_a.point0[0] > cinfo_b.point1[0]:
        tmp = cinfo_a
        cinfo_a = cinfo_b
        cinfo_b = tmp
    x_overlap_a = cinfo_a.local_overlap(cinfo_b)
    x_overlap_b = cinfo_b.local_overlap(cinfo_a)
    overall_tangent = cinfo_b.center - cinfo_a.center
    overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])
    delta_angle = max(angle_dist(cinfo_a.angle, overall_angle),
                      angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi
    x_overlap = max(x_overlap_a, x_overlap_b)
    dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)
    if (dist > EDGE_MAX_LENGTH or
            x_overlap > EDGE_MAX_OVERLAP or
            delta_angle > EDGE_MAX_ANGLE):
        return None
    else:
        score = dist + delta_angle*EDGE_ANGLE_COST
        return (score, cinfo_a, cinfo_b)

def make_tight_mask(contour, xmin, ymin, width, height):
    tight_mask = np.zeros((height, width), dtype=np.uint8)
    tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))
    cv2.drawContours(tight_mask, [tight_contour], 0,
                     (1, 1, 1), -1)
    return tight_mask

def get_contours(small, pagemask):
    mask = get_mask(small, pagemask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contours_out = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        xmin, ymin, width, height = rect
        if (width < TEXT_MIN_WIDTH or
                height < TEXT_MIN_HEIGHT or
                width < TEXT_MIN_ASPECT*height):
            continue
        tight_mask = make_tight_mask(contour, xmin, ymin, width, height)
        if tight_mask.sum(axis=0).max() > TEXT_MAX_THICKNESS:
            continue
        contours_out.append(ContourInfo(contour, rect, tight_mask))
    return contours_out

def assemble_spans(cinfo_list):
    cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])
    candidate_edges = []
    for i, cinfo_i in enumerate(cinfo_list):
        for j in range(i):
            edge = generate_candidate_edge(cinfo_i, cinfo_list[j])
            if edge is not None:
                candidate_edges.append(edge)
    candidate_edges.sort()
    for _, cinfo_a, cinfo_b in candidate_edges:
        if cinfo_a.succ is None and cinfo_b.pred is None:
            cinfo_a.succ = cinfo_b
            cinfo_b.pred = cinfo_a
    spans = []
    while cinfo_list:
        cinfo = cinfo_list[0]
        while cinfo.pred:
            cinfo = cinfo.pred
        cur_span = []
        width = 0.0
        while cinfo:
            cinfo_list.remove(cinfo)
            cur_span.append(cinfo)
            width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
            cinfo = cinfo.succ
        if width > SPAN_MIN_WIDTH:
            spans.append(cur_span)
    return spans

def sample_spans(shape, spans):
    span_points = []
    for span in spans:
        contour_points = []
        for cinfo in span:
            yvals = np.arange(cinfo.mask.shape[0]).reshape(
                (-1, 1))  # mask의 y좌표
            totals = (yvals * cinfo.mask).sum(axis=0)
            means = totals / cinfo.mask.sum(axis=0)  # mask y축의 중간 값
            xmin, ymin = cinfo.rect[:2]  # 마스크의 좌측 상단 좌표
            step = SPAN_PX_PER_STEP  # 20
            start = ((len(means)-1) % step) / 2
            contour_points += [(x+xmin, means[x]+ymin)
                               for x in range(int(start), len(means), step)]  # 마스크에 두께 중간의 좌표 20픽셀마다 점으로 넣어줌
        contour_points = np.array(contour_points,
                                  dtype=np.float32).reshape((-1, 1, 2))
        contour_points = pix2norm(shape, contour_points)
        span_points.append(contour_points)
    return span_points

def keypoints_from_samples(pagemask, page_outline, span_points):
    all_evecs = np.array([[0.0, 0.0]])
    all_weights = 0
    for points in span_points:
        _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                 None, maxComponents=1)
        weight = np.linalg.norm(points[-1] - points[0])
        all_evecs += evec * weight
        all_weights += weight
    evec = all_evecs / all_weights
    x_dir = evec.flatten()
    if x_dir[0] < 0:
        x_dir = -x_dir
    y_dir = np.array([-x_dir[1], x_dir[0]])
    pagecoords = cv2.convexHull(page_outline)
    pagecoords = pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
    pagecoords = pagecoords.reshape((-1, 2))
    px_coords = np.dot(pagecoords, x_dir)
    py_coords = np.dot(pagecoords, y_dir)
    px0 = px_coords.min()
    px1 = px_coords.max()
    py0 = py_coords.min()
    py1 = py_coords.max()
    p00 = px0 * x_dir + py0 * y_dir
    p10 = px1 * x_dir + py0 * y_dir
    p11 = px1 * x_dir + py1 * y_dir
    p01 = px0 * x_dir + py1 * y_dir
    corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))
    ycoords = []
    xcoords = []
    for points in span_points:
        pts = points.reshape((-1, 2))
        px_coords = np.dot(pts, x_dir)
        py_coords = np.dot(pts, y_dir)
        ycoords.append(py_coords.mean() - py0)
        xcoords.append(px_coords - px0)
    return corners, np.array(ycoords), xcoords

def make_keypoint_index(span_counts):
    nspans = len(span_counts)
    npts = sum(span_counts)
    keypoint_index = np.zeros((npts+1, 2), dtype=int)
    start = 1
    for i, count in enumerate(span_counts):
        end = start + count
        keypoint_index[start:start+end, 1] = 8+i
        start = end
    keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans
    return keypoint_index


def optimize_params(small, dstpoints, span_counts, params):
    keypoint_index = make_keypoint_index(span_counts)
    def objective(pvec):
        ppts = project_keypoints(pvec, keypoint_index)
        return np.sum((dstpoints - ppts)**2)
    res = scipy.optimize.minimize(objective, params,
                                  method='Powell')
    params = res.x
    return params

def get_page_dims(corners, rough_dims, params):
    dst_br = corners[2].flatten()
    dims = np.array(rough_dims)
    def objective(dims):
        proj_br = project_xy(dims, params)
        return np.sum((dst_br - proj_br.flatten())**2)
    res = scipy.optimize.minimize(objective, dims, method='Powell')
    dims = res.x
    return dims

def remap_image(img, page_dims, params):
    height = 0.5 * page_dims[1] * OUTPUT_ZOOM * img.shape[0]
    height = round_nearest_multiple(height, REMAP_DECIMATE)
    width = round_nearest_multiple(height * page_dims[0] / page_dims[1],
                                   REMAP_DECIMATE)
    height_small = int(height / REMAP_DECIMATE)
    width_small = int(width / REMAP_DECIMATE)
    page_x_range = np.linspace(0, page_dims[0], width_small)
    page_y_range = np.linspace(0, page_dims[1], height_small)
    page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)
    page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                page_y_coords.flatten().reshape((-1, 1))))
    page_xy_coords = page_xy_coords.astype(np.float32)
    image_points = project_xy(page_xy_coords, params)
    image_points = norm2pix(img.shape, image_points, False)
    image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
    image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)
    image_x_coords = cv2.resize(image_x_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)
    image_y_coords = cv2.resize(image_y_coords, (width, height),
                                interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    remapped = cv2.remap(img_gray, image_x_coords, image_y_coords,
                         cv2.INTER_CUBIC,
                         None, cv2.BORDER_REPLICATE)
    thresh = cv2.adaptiveThreshold(remapped, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, ADAPTIVE_WINSZ, 25)

    return thresh

def run(inputimage):
    img = inputimage
    small = resize_to_screen(img)
    pagemask, page_outline = get_page_extents(small)
    print('get mask')
    cinfo_list = get_contours(small, pagemask)
    print('get contours')
    spans = assemble_spans(cinfo_list)
    span_points = sample_spans(small.shape, spans)  # 이미지에서 실제 좌표는 아님
    print('get sample')
    corners, ycoords, xcoords = keypoints_from_samples(pagemask,
                                                       page_outline,
                                                       span_points)
    rough_dims, span_counts, params = get_default_params(corners,
                                                         ycoords, xcoords)
    print('get parmas')
    dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
                          tuple(span_points))
    print('doing dewarp')
    params = optimize_params(small,
                             dstpoints,
                             span_counts, params)
    page_dims = get_page_dims(corners, rough_dims, params)
    print('doing remap')
    outfile = remap_image(img, page_dims, params)
    return outfile

