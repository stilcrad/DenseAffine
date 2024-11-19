import os

import numpy as np
import cv2

def get_optimal_affine_transformation(A, F, pt1, pt2):
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)

    # 检查输入数据
    assert pt1.shape[1] == 2 or pt1.shape[1] == 3
    assert pt1.shape == pt2.shape
    assert A.shape[1] == 2 and A.shape[0] == 2
    assert A.dtype == F.dtype

    # 计算极线
    l1 = np.dot(F.T, pt2.T)
    l2 = np.dot(F, pt1.T)

    l1 = l1 / l1[2]
    l2 = l2 / l2[2]

    n1 = np.array([l1[0], l1[1]])
    n2 = np.array([l2[0], l2[1]])
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # 计算尺度参数 beta
    beta = get_beta_scale(A, F, pt1, pt2)

    if np.dot(n1.transpose(), n2) < 0:
        n2 = -n2

    # 计算最优仿射变换
    C = np.array([[1, 0, 0, 0, -beta[0] * n2[0][0], 0],
                  [0, 1, 0, 0, 0, -beta[0] * n2[0][0]],
                  [0, 0, 1, 0, -beta[0] * n2[1][0], 0],
                  [0, 0, 0, 1, 0, -beta[0] * n2[1][0]],
                  [-beta[0] * n2[0][0], 0, -beta[0] * n2[1][0], 0, 0, 0],
                  [0, -beta[0] * n2[0][0], 0, -beta[0] * n2[1][0], 0, 0]],dtype=np.float64)

    b = np.array([A[0, 0], A[0, 1], A[1, 0], A[1, 1], -n1[0][0], -n1[1][0]])
    x = np.linalg.inv(C).dot(b)

    opt_A = np.array([[x[0], x[1]], [x[2], x[3]]])

    return opt_A


def get_beta_scale(A, F, pt1, pt2):
    pt1 = np.asarray(pt1)
    pt2 = np.asarray(pt2)

    # 检查输入数据
    assert pt1.shape[1] == 2 or pt1.shape[1] == 3
    assert pt1.shape == pt2.shape
    assert A.shape[1] == 2 and A.shape[0] == 2
    assert A.dtype == F.dtype

    if pt1.shape[0] == 1:
        pt1 = pt1.T
        pt2 = pt2.T

    if pt1.shape[1] == 2:
        pt1 = np.vstack((pt1.T, np.ones((1, pt1.shape[0]))))
        pt2 = np.vstack((pt2.T, np.ones((1, pt2.shape[0]))))

    # 计算极线
    l1 = np.dot(F.T, pt2)
    l2 = np.dot(F, pt1)

    x_new1 = pt1[0] + 1.0
    y_new1 = -(l1[0] * x_new1 + l1[2]) / l1[1]
    dx1 = np.array([x_new1[0], y_new1[0], 1]) - pt1.T

    x_new2 = pt2[0] + 1.0
    y_new2 = -(l2[0] * x_new2 + l2[2]) / l2[1]
    dx2 = np.array([x_new2[0], y_new2[0], 1]) - pt2.T

    dx1 = dx1 / np.linalg.norm(dx1)
    dx2 = dx2 / np.linalg.norm(dx2)

    beta = abs(np.sqrt(l2[0] * l2[0] + l2[1] * l2[1]) /
               ((-F[0, 0] * dx1[0][1] + F[0, 1] * dx1[0][0]) * pt2[0] +
                (-F[1, 0] * dx1[0][1] + F[1, 1] * dx1[0][0]) * pt2[1] - F[2, 0] * dx1[0][1] + F[2, 1] * dx1[0][0]))

    return beta

#matlab fanyi
import numpy as np


def HartleySturmTriangulation(F, u1, v1):
    T1 = np.array([[1, 0, -u1[0]], [0, 1, -u1[1]], [0, 0, 1]])
    T2 = np.array([[1, 0, -v1[0]], [0, 1, -v1[1]], [0, 0, 1]])

    F2 = np.linalg.inv(T2.T) @ F @ np.linalg.inv(T1)

    e1 = np.linalg.null_space(F2)
    e2 = np.linalg.null_space(F2.T)

    s1 = e1[0] ** 2 + e1[1] ** 2
    s2 = e2[0] ** 2 + e2[1] ** 2

    R1 = np.array([[e1[0], e1[1], 0], [-e1[1], e1[0], 0], [0, 0, 1]])
    R2 = np.array([[-e2[0], -e2[1], 0], [e2[1], -e2[0], 0], [0, 0, 1]])

    F3 = R2 @ F2 @ R1.T

    f1 = e1[2]
    f2 = e2[2]

    a = F3[1, 1]
    b = F3[1, 2]
    c = F3[2, 1]
    d = F3[2, 2]

    t6 = -a * c * (f1 ** 4) * (a * d - b * c)
    t5 = (a ** 2 + f2 ** 2 * c ** 2) ** 2 - (a * d + b * c) * (f1 ** 4) * (a * d - b * c)
    t4 = 2 * (a ** 2 + f2 ** 2 * c ** 2) * (2 * a * b + 2 * c * d * f2 ** 2) - d * b * (f1 ** 4) * (
                a * d - b * c) - 2 * a * c * f1 ** 2 * (a * d - b * c)
    t3 = (2 * a * b + 2 * c * d * f2 ** 2) ** 2 + 2 * (a ** 2 + f2 ** 2 * c ** 2) * (
                b ** 2 + f2 ** 2 * d ** 2) - 2 * f1 ** 2 * (a * d - b * c) * (a * d + b * c)
    t2 = 2 * (2 * a * b + 2 * c * d * f2 ** 2) * (b ** 2 + f2 ** 2 * d ** 2) - 2 * (
                f1 ** 2 * a * d - f1 ** 2 * b * c) * b * d - a * c * (a * d - b * c)
    t1 = (b ** 2 + f2 ** 2 * d ** 2) ** 2 - (a * d + b * c) * (a * d - b * c)
    t0 = -(a * d - b * c) * b * d

    r = [t6, t5, t4, t3, t2, t1, t0]

    bestS = np.inf
    rs = np.roots(r)

    for i in range(rs.shape[0]):
        currRoot = rs[i]
        if np.isreal(currRoot):
            val = calculateS(currRoot, a, b, c, d, f1, f2)
            if val < bestS:
                bestS = val
                bestT = currRoot

    valInf = 1 / (f1 ** 2) + (c ** 2) / (a ** 2 + f2 ** 2 * c ** 2)
    if valInf < bestS:
        print('Serious error')
        exit()

    point1 = np.array([0, bestT, 1])
    line2 = F3 @ point1
    point2 = np.array([-line2[0] * line2[2], -line2[1] * line2[2], line2[0] ** 2 + line2[1] ** 2])
    point2 = point2 / point2[2]

    u2 = np.linalg.inv(R1 @ T1) @ point1
    v2 = np.linalg.inv(R2 @ T2) @ point2

    return u2, v2

def EGL2OptimalAffineCorrection(affine, F, pt1, pt2):
    p1, p2 = HartleySturmTriangulation(F, pt1, pt2)

    l2 = np.dot(F, p1)
    l1 = np.dot(F.T, p2)

    p1 = p1[:2]
    p2 = p2[:2]

    x_new1 = p1[0] + 1.0
    y_new1 = -(l1[0] * x_new1 + l1[2]) / l1[1]
    dx1 = np.array([x_new1, y_new1]) - p1

    x_new2 = p2[0] + 1.0
    y_new2 = -(l2[0] * x_new2 + l2[2]) / l2[1]
    dx2 = np.array([x_new2, y_new2]) - p2

    dx1 = dx1 / np.linalg.norm(dx1)
    dx2 = dx2 / np.linalg.norm(dx2)

    if np.dot(dx2.T, np.dot(affine, dx1)) < 0:
        dx2 = -dx2

    beta = np.sqrt(l2[0]*l2[0] + l2[1]*l2[1]) / ((-F[0,0]*dx1[1] + F[0,1]*dx1[0])*p2[0] +
                                                 (-F[1,0]*dx1[1] + F[1,1]*dx1[0])*p2[1] - F[2,0]*dx1[1] + F[2,1]*dx1[0])
    beta = abs(beta)
    beta = 1.0 / beta

    n1 = np.array([-dx1[1], dx1[0]])
    n2 = np.array([-dx2[1], dx2[0]])

    CFS = np.zeros((6, 6))
    CFS[0, :] = [1, 0, 0, 0, n2[0]*dx1[0], n2[0]*n1[0]]
    CFS[1, :] = [0, 1, 0, 0, n2[0]*dx1[1], n2[0]*n1[1]]
    CFS[2, :] = [0, 0, 1, 0, n2[1]*dx1[0], n2[1]*n1[0]]
    CFS[3, :] = [0, 0, 0, 1, n2[1]*dx1[1], n2[1]*n1[1]]
    CFS[4, :] = [n2[0]*dx1[0], n2[0]*dx1[1], n2[1]*dx1[0], n2[1]*dx1[1], 0, 0]
    CFS[5, :] = [n2[0]*n1[0], n2[0]*n1[1], n2[1]*n1[0], n2[1]*n1[1], 0, 0]

    bs1 = np.array([affine[0, 0], affine[0, 1], affine[1, 0], affine[1, 1], 0.0, beta])
    bs2 = np.array([affine[0, 0], affine[0, 1], affine[1, 0], affine[1, 1], 0.0, -beta])

    Aest1 = np.dot(np.linalg.inv(CFS), bs1)
    Aest2 = np.dot(np.linalg.inv(CFS), bs2)

    A1 = np.zeros((2, 2))
    A1[0, 0] = Aest1[0]
    A1[0, 1] = Aest1[1]
    A1[1, 0] = Aest1[2]
    A1[1, 1] = Aest1[3]

    A2 = np.zeros((2, 2))
    A2[0, 0] = Aest2[0]
    A2[0, 1] = Aest2[1]
    A2[1, 0] = Aest2[2]
    A2[1, 1] = Aest2[3]

    if np.linalg.norm(A1 - affine) < np.linalg.norm(A2 - affine):
        A = A1
    else:
        A = A2

    return A


# 生成齐次坐标点
def generate_homogeneous_points(num_points):
    points = np.random.rand(num_points, 2)  # 生成随机的二维坐标点
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))  # 在每个点后面添加一个齐次坐标1
    return homogeneous_points

# 生成10个齐次坐标点
homogeneous_points = generate_homogeneous_points(10)

if __name__ == '__main__':


    for i in range(len(os.listdir("/home/xxx/project/python/DKM-main/affine/out_ac/kitti_1/new_ACs_04/"))):

        with open(
                "/home/xxx/project/python/DKM-main/affine/out_ac/kitti_1/new_ACs_04/outA_new{}.txt".format(
                    i), "r") as f1:
            the_ac = f1.readlines()
            ACs_DKM = np.array([np.array([float(i) for i in j.split(",")[:8]]) for j in the_ac])
            dkm_pt1 = ACs_DKM[:, 0:2]
            dkm_pt2 = ACs_DKM[:, 2:4]
            affines = ACs_DKM[:, 4:8]
            F , _=cv2.findFundamentalMat(dkm_pt1,dkm_pt2,method=cv2.USAC_ACCURATE)
            with open(
                    "/home/xxx/project/python/DKM-main/affine/out_ac/kitti_1/OP_new_ACs_04/outA_new{}.txt".format(
                        i), "w") as f2:
                for j in range(dkm_pt1.shape[0]):
                   pt1 = np.hstack((dkm_pt1[j], [1]))
                   pt2 = np.hstack((dkm_pt2[j], [1]))
                   A = affines[j]
                   optimal_affine = EGL2OptimalAffineCorrection(A, F, pt1, pt2)

                   acrxt = list(str(m) for m in list(pt1) + list(pt2) + list(optimal_affine))
                   f2.write(' '.join(acrxt))
                   f2.write("\n")

