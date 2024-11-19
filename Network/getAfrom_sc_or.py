import numpy as np
import cmath


def GetA(alpha, beta, scale1 ,scale2 , pt1, pt2,F):
    scale = scale2/scale1
    ca = cmath.cos(beta)
    sa = cmath.sin(beta)
    cb = cmath.cos(alpha)
    sb = cmath.sin(alpha)

    u1 = pt1[0]
    v1 = pt1[1]
    u2 = pt2[0]
    v2 = pt2[1]

    f1 = F[0][0]
    f2 = F[0][1]
    f3 = F[0][2]
    f4 = F[1][0]
    f5 = F[1][1]
    f6 = F[1][2]
    f7 = F[2][0]
    f8 = F[2][1]
    f9 = F[2][2]

    A1 = (ca * cb * u1 * f1 + ca * cb * v1 * f2 + ca * cb * f3 + sa * cb * u1 * f4 + sa * cb * v1 * f5 + sa * cb * f6)
    B1 = (-sa * sb * u1 * f1 - sa * sb * v1 * f2 - sa * sb * f3 + ca * sb * f4 * u1 + ca * sb * v1 * f5 + ca * sb * f6)
    C1 = (ca * sb * u1 * f1 + ca * sb * v1 * f2 + ca * sb * f3 + sa * sb * f4 * u1 + sa * sb * v1 * f5 + sa * sb * f6)
    D1 = u2 * f1 + v2 * f4 + f7

    A2 = (- ca * sb * u1 * f1 - ca * sb * v1 * f2 - ca * sb * f3 - sa * sb * u1 * f4 - sa * sb * f6 - sa * sb * v1 * f5)
    B2 = (- sa * cb * u1 * f1 - sa * cb * v1 * f2 - sa * cb * f3 + ca * cb * u1 * f4 + ca * cb * f6 + ca * cb * v1 * f5)
    C2 = (ca * cb * u1 * f1 + ca * cb * v1 * f2 + ca * cb * f3 + sa * cb * u1 * f4 + sa * cb * f6 + sa * cb * v1 * f5)
    D2 = u2 * f2 + v2 * f5 + f8

    a = (B2 - C2 * B1 / C1)
    b = (D2 - C2 * D1 / C1)
    c = scale * (A2 - C2 * A1 / C1)

    if cmath.isinf(a) or cmath.isnan(a) or cmath.isinf(b) or cmath.isnan(b) or cmath.isinf(c) or cmath.isnan(c):
        A = 0
        return A

    r = np.roots([a, b, c])

    A = []
    best_A = 0
    best_err = 1e10

    for ri in range(len(r)):
        qvi = r[ri]
        if abs(a * qvi ** 2 + b * qvi + c) > 1e-4:
            continue

        qui = scale / qvi
        wi = -A1 / C1 * qui - B1 / C1 * qvi - D1 / C1

        Ai = np.dot(np.array([[cmath.cos(beta), -cmath.sin(beta)], [cmath.sin(beta), cmath.cos(beta)]]),
                    np.array([[qui, wi], [0, qvi]]),
                    np.array([[cmath.cos(alpha), -cmath.sin(alpha)], [cmath.sin(alpha), cmath.cos(alpha)]]))
        A.append(Ai)

    return A

def get_affine_from_just_sca_rot(alpha_1_left, alpha_2_left,scale1,scale2):
    Ac_c1_all=[]
    R_1_left = np.array([[np.cos(alpha_1_left), -np.sin(alpha_1_left)], [np.sin(alpha_1_left), np.cos(alpha_1_left)]])

    S_1_left = np.array([[scale1, 0], [0, scale1]])


    R_2_left = np.array([[np.cos(alpha_2_left), -np.sin(alpha_2_left)], [np.sin(alpha_2_left), np.cos(alpha_2_left)]])

    S_2_left = np.array([[scale2, 0], [0, scale2]])

    Ac_c1 = np.matmul(np.hstack((R_2_left, S_2_left)), np.linalg.pinv(np.hstack((R_1_left, S_1_left))))

    Ac_c1_all.append([Ac_c1[0, 0], Ac_c1[0, 1], Ac_c1[1, 0], Ac_c1[1, 1]])

    return Ac_c1



if __name__ == '__main__':

    alpha = np.pi / 4  # 45 degrees
    beta = np.pi / 6   # 30 degrees
    scale1 = 1
    scale2 = 2

    pt1 = [224.214, 242.7]
    pt2 = [141.1, 540.1]

    F = np.array([[0.141, 0.127425, 0.247274],
                  [0.7412,0.4424,0.424],
                  [0.424, 0.4242, 0.4527425]])

    A = GetA(alpha, beta, scale1,scale2, pt1, pt2, F,)


    A2 = get_affine_from_just_sca_rot(alpha, beta, scale1, scale2)


    print(A)
    print(A2)