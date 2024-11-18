import cv2
import numpy as np
import os

def drawMatches(img1, kp1, img2, kp2, matches, matchColor):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])

    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates

        # thickness = 1
        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 256)

        cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, matchColor, 1)  # 
        cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, matchColor, 1)

        # Draw a line in between the two points
        # colour blue
        cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))),
                 matchColor, 1, shift=0)  # 

    # Also return the image if you'd like a copy
    return out



def match_image(path1 ,path2,ac_path , out_path):
    # Load images
    img1 = cv2.imread(path1)  # queryImage
    img2 = cv2.imread(path2)  # trainImage

    space_len = 5
    # Load matching points
    b = []
    with open(ac_path) as f :
        line = f.readlines()
        for i in line:
            b.append([int(float(j)) for j in i.strip().split(",")][:4])

        kp_np = np.array(b)[1:2000]

    matching_points = kp_np
    if matching_points.shape[0] == 0:
        return 0 , 0
    # matching_points = kp_np

    # Create keypoint objects
    keypoints1 = [cv2.KeyPoint(float(point[0]), float(point[1]), 1) for point in matching_points[:, :2]]
    keypoints2 = [cv2.KeyPoint(float(point[0]), float(point[1]), 1) for point in matching_points[:, 2:]]

    # Load homography matrix from file


    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((rows1 + rows2 + 3, max([cols1, cols2]), 3), dtype='uint8')
    out[:,:] = [255, 255, 255]

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1])
    # out[:rows1,cols1:cols1+space_len] =[255, 255, 255]

    # Place the next image to the right of it
    out[rows1 + 3:rows1 + rows2 + 3, :cols2] = np.dstack([img2])
    matchColor = []
    dist_all = []
    num = 0
    for i, (kp1, kp2) in enumerate(zip(keypoints1, keypoints2)):
        if i%5 ==0 :
            matchColor = (0,255,0)
            cv2.circle(out, (int(np.round(kp1.pt[0])), int(np.round(kp1.pt[1]))), 2, matchColor, 1)  #
            cv2.circle(out, (int(np.round(kp2.pt[0]) ), int(np.round(kp2.pt[1]))+ rows1 +space_len), 2, matchColor, 1)
            cv2.line(out, (int(np.round(kp1.pt[0])), int(np.round(kp1.pt[1]))), (int(np.round(kp2.pt[0])), int(np.round(kp2.pt[1])) + rows1+space_len), matchColor, 1,shift=0)

    cv2.imwrite(out_path, out)
    cv2.imshow("matches",out)
    cv2.waitKey(5000)

    return np.array(dist_all), num


if __name__ == '__main__':

    path1 = r"E:\EVD\1\grand.png"
    path2 = r"E:\EVD\2\grand.png"
    ac_path = r"E:\EVD\ours_ACs\grand_ACs.txt"
    H_path = r"E:\EVD\h\grand.txt"
    dist_all, num = match_image(path1, path2, ac_path,H_path,"grand")


