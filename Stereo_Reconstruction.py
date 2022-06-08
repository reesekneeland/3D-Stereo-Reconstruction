import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

def find_match(img1, img2):
    xim1 = []
    yim1 = []
    xim2 = []
    yim2 = []
    x = []
    y = []
    s1 = cv2.xfeatures2d.SIFT_create()
    s2 = cv2.xfeatures2d.SIFT_create()
    keyp1, desc1 = s1.detectAndCompute(img1, None)
    keyp2, desc2 = s2.detectAndCompute(img2, None)
    neighbor1 = NearestNeighbors()
    neighbor1.fit(desc2)
    dist1, ind1 = neighbor1.kneighbors(desc1, n_neighbors=2)
    neighbor2 = NearestNeighbors()
    neighbor2.fit(desc1)
    dist2, ind2 = neighbor2.kneighbors(desc2, n_neighbors=2)
    
    for i in range(len(dist1)):
        if (dist1[i,0] / dist1[i,1]) < 0.75:
            xim1.append(keyp1[i].pt)
            yim1.append(keyp2[ind1[i,0]].pt)
    for i in range(len(dist2)):
        if (dist2[i,0] / dist2[i,1]) < 0.75:
            xim2.append(keyp2[i].pt)
            yim2.append(keyp1[ind2[i,0]].pt)
    for i in range(len(xim1)): 
        for j in range(len(yim2)):
            if (xim1[i][0] == yim2[j][0] and xim1[i][1] == yim2[j][1] and yim1[i][0] == xim2[j][0] and yim1[i][1] == xim2[j][1]):
                    x.append(xim1[i])
                    y.append(yim1[i])
    pts1 = np.asarray(x)
    pts2 = np.asarray(y)
    return pts1, pts2


def compute_F(pts1, pts2):
    iters = 1000
    iter=0
    thresh = 0.025
    maxIl = 0
    numPoints = len(pts1)
    numSamples = 8
    while iter < iters:
        iter+=1
        a=np.zeros((numSamples,9))
        sampler = np.random.choice(numPoints,numSamples)
        random1 = pts1[sampler,:]
        random2 = pts2[sampler,:]
        np.append(random1, [[1]]*numSamples, axis=1)
        np.append(random2, [[1]]*numSamples, axis=1)

        for i in range(numSamples):
            a[i] = np.array([random1[i][0]*random2[i][0], random1[i][1]*random2[i][0], random2[i][0], random1[i][0]*random2[i][1], random1[i][1]*random2[i][1], random2[i][1], random1[i][0], random1[i][1], 1])
        fnull = null_space(a)
        fnull = fnull[:,0].reshape(3,3)
        u, d, vt = np.linalg.svd(fnull)
        d[-1] = 0
        fcur = (u @ np.diag(d)) @ vt

        il = 0
        for i in range(numPoints):
            point1 = np.transpose(np.array([pts1[i][0], pts1[i][1], 1]))
            point2 = np.array([pts2[i][0], pts2[i][1], 1])
            err = abs((point2 @ fcur) @ point1)
            if(err < thresh):
                il+=1 
        if(il > maxIl):
            maxIl = il
            F=fcur
    return F

def skew_pt(pt):
    return np.array([[0,-pt[2],pt[1]], [pt[2],0,-pt[0]], [-pt[1],pt[0],0]])

def triangulation(P1, P2, pts1, pts2):
    pts3D=np.zeros((len(pts1), 3))
    for i in range(len(pts1)):
        pt13d = list(pts1[i,:]) + [1]
        pt23d = list(pts2[i,:]) + [1]
        p13dskewed = skew_pt(pt13d) @ P1
        p23dskewed = skew_pt(pt23d) @ P2
        u, s, vt = np.linalg.svd(np.concatenate((p13dskewed[0:2,:], p23dskewed[0:2,:])))
        v = vt.T
        pt = v[:,-1]
        scaled = pt[:3] / pt[3]
        pts3D[i] = scaled
    return pts3D

def disambiguate_pose(Rs, Cs, pts3Ds):
    mSum = 0
    id_max = 0
    for i in range(len(Rs)):
        n = 0
        curRotation = np.array(Rs[i])[2:]
        curCenter = np.reshape(np.array(Cs[i]),-1)
        for j in range(len(pts3Ds[0])):
            applied = (pts3Ds[i][j,:] - curCenter) @  curRotation[0].T
            if applied > 0: n+=1
        if n > mSum:
            mSum=n
            id_max = i 
    return Rs[id_max], Cs[id_max], pts3Ds[id_max]

def compute_rectification(K, R, C):
    C = C.reshape((1,3))
    x = C/np.linalg.norm(C)
    zt = [0,0,1]
    zt = np.array(zt)
    el1 = zt @ x.T
    el2 = el1 @ x
    el3 = zt - el2
    el4 = np.linalg.norm(el3)
    z = el3/el4
    z = z.reshape((1,3))
    y = np.cross(z, x)
    rectangle = np.array([x, y, z]).reshape(3,3)
    inv = np.linalg.inv(K)
    H1 = K @ rectangle @ inv
    H2 = K @ rectangle @ R.T @ inv

    return H1, H2

def dense_match(img1, img2):
    disparity = np.zeros(np.shape(img1))
    sift1 = cv2.xfeatures2d.SIFT_create()
    sift2 = cv2.xfeatures2d.SIFT_create()
    keyp1 = [cv2.KeyPoint(i, j, 3) for j in range(0, len(img1)) 
                                         for i in range(0, len(img1[0]))]
    keyp_comp1, feat1 = sift1.compute(img1, keyp1)
    keyp2 = [cv2.KeyPoint(i, j, 3) for j in range(0, len(img2)) 
                                         for i in range(0, len(img2[0]))]
    keyp_comp2, feat2 = sift2.compute(img2, keyp2)
    keyp3 = [cv2.KeyPoint(i, j, 10) for j in range(0, len(img1)) 
                                         for i in range(0, len(img1[0]))]
    keyp_comp3, feat3 = sift1.compute(img1, keyp3)
    keyp4 = [cv2.KeyPoint(i, j, 10) for j in range(0, len(img2)) 
                                         for i in range(0, len(img2[0]))]
    keyp_comp4, feat4 = sift2.compute(img2, keyp4)
    feat1 = feat1.reshape((len(img1), len(img1[0]), 128))
    feat2 = feat2.reshape((len(img2), len(img2[0]), 128))
    feat3 = feat3.reshape((len(img1), len(img1[0]), 128))
    feat4 = feat4.reshape((len(img2), len(img2[0]), 128))
    for i in range(len(img1)):
        for j in range(np.shape(img1)[1]):  
            if img1[i,j] != 0:
                ranges = []
                ranges2 = []
                curFeat1 = feat1[i, j]
                curFeat3 = feat3[i, j]
                for curJ in range(len(img1[0])):
                    curFeat2 = feat2[i, curJ]
                    curFeat4 = feat4[i, curJ]
                    dist = np.linalg.norm(curFeat1 - curFeat2)
                    dist2 = np.linalg.norm(curFeat3 - curFeat4)
                    ranges.append(dist)
                    ranges2.append(dist2)
                smallestDist = np.argmin(ranges)
                smallestDist2 = np.argmin(ranges2)
                dist1 = np.abs(smallestDist - j)
                dist2 = np.abs(smallestDist2 - j)
                disparity[i,j] = dist1+(3*dist2)
    return disparity



# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()

#had to cast to ints to get this function to work
def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, int((-el[2] / el[1])[0])), (img.shape[1], int(((-img_width * el[0] - el[2]) / el[1])[0]))
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})