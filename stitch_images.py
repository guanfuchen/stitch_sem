import glob
import os
import sys
import cv2
import numpy as np

# Use the keypoints to stitch the images
def get_stitched_image(img1, img2, M):

    # Get width and height of input images
    w1,h1 = img1.shape[:2]
    w2,h2 = img2.shape[:2]

    # Get the canvas dimesions
    img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
    img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)


    # Get relative perspective of second image
    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    # Resulting dimensions
    result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

    # Getting images together
    # Calculate dimensions of match points
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    # Create output array after affine transformation
    transform_dist = [-x_min,-y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0,0,1]])

    # Warp images to get the resulting image
    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                    (x_max-x_min, y_max-y_min))
    result_img[transform_dist[1]:w1+transform_dist[1],
                transform_dist[0]:h1+transform_dist[0]] = img1

    # Return the result
    return result_img

# Find SIFT and return Homography Matrix
def get_sift_homography(img1, img2):

    # Initialize SIFT
    sift = cv2.SIFT()

    # Extract keypoints and descriptors
    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    # Bruteforce matcher on the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1,d2, k=2)

    # Make sure that the matches are good
    verify_ratio = 0.8 # Source: stackoverflow
    verified_matches = []
    for m1,m2 in matches:
        # Add to array only if it's a good match
        if m1.distance < 0.8 * m2.distance:
            verified_matches.append(m1)

    # Mimnum number of matches
    min_matches = 8
    if len(verified_matches) > min_matches:

        # Array to store matching points
        img1_pts = []
        img2_pts = []

        # Add matching points to array
        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1,1,2)
        img2_pts = np.float32(img2_pts).reshape(-1,1,2)

        # Compute homography matrix
        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        print 'Error: Not enough matches'
        return None
        # exit()

# Equalize Histogram of Color Images
def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

# Main function definition
def main():

    # Get input set of images
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])

    img1_label = None
    img2_label = None
    if len(sys.argv) == 5:
        img1_label = cv2.imread(sys.argv[3])
        img2_label = cv2.imread(sys.argv[4])
    # print('img1_label:', img1_label)
    # print('img2_label:', img2_label)

    # Equalize histogram
    img1 = equalize_histogram_color(img1)
    img2 = equalize_histogram_color(img2)

    # Show input images
    #input_images = np.hstack( (img1, img2) )
    #cv2.imshow ('Input Images', input_images)

    # Use SIFT to find keypoints and return homography matrix
    M =  get_sift_homography(img1, img2)

    # Stitch the images together using homography matrix

    result_image = get_stitched_image(img2, img1, M)
    result_image_label = None
    if img1_label is not None and img2_label is not None:
        print("result_image_label")
        result_image_label = get_stitched_image(img2_label, img1_label, M)

    # Write the result to the same directory
    # result_image_name = 'results/result_'+sys.argv[1]
    # cv2.imwrite(result_image_name, result_image)

    # Show the resulting image
    cv2.imshow ('Result', result_image)
    if result_image_label is not None:
        cv2.imshow ('Result_Label', result_image_label)
    cv2.waitKey()

def main1():
    # print(sys.argv)
    file_prefix = 'bmp'
    images = glob.glob(sys.argv[1]+'*.{}'.format(file_prefix))
    if len(sys.argv) == 3:
        labels = glob.glob(sys.argv[2]+'*.{}'.format(file_prefix))
    elif len(sys.argv) == 2:
        labels = glob.glob(sys.argv[1]+'*.{}'.format(file_prefix))
    images.sort(reverse=False)
    labels.sort(reverse=False)

    # print(images)
    # print(labels)
    prev_image = cv2.imread(images[0])
    prev_label = cv2.imread(labels[0])

    prev_image = equalize_histogram_color(prev_image)
    # prev_label = equalize_histogram_color(prev_label)
    Ms = []
    for image_id, image in enumerate(images[1:]):
        # print(image_id)
        curr_image = cv2.imread(image)
        curr_label = cv2.imread(labels[image_id])
        curr_image = equalize_histogram_color(curr_image)
        # curr_label = equalize_histogram_color(curr_label)
        M = get_sift_homography(prev_image, curr_image)
        # print(M)
        Ms.append(M)
        # print(M)
        if M is not None:
            # prev_image = curr_image
            # prev_label = curr_label
            prev_image = get_stitched_image(curr_image, prev_image, M)
            prev_label = get_stitched_image(curr_label, prev_label, M)

    # M_prev = Ms[0]
    # for M_value in Ms[1:]:
    #     M_prev = np.dot(M_prev, M_value)
    #     print(M_prev)
    # reverse_images = images[::-1]
    # reverse_labels = labels[::-1]
    # reverse_images = images
    # reverse_labels = labels

    # print(images)
    # print(labels)
    # print(reverse_images)
    # print(reverse_labels)
    # curr_image = cv2.imread(reverse_images[0])
    # curr_label = cv2.imread(reverse_labels[0])
    # curr_image = equalize_histogram_color(curr_image)
    # M_prev = np.ones((3, 3))
    # for image_id, image in enumerate(reverse_images[1:]):
    #     prev_image = cv2.imread(image)
    #     prev_label = cv2.imread(labels[image_id])
    #     prev_image = equalize_histogram_color(prev_image)
    #     if Ms[image_id] is not None:
    #         M_prev = np.matmul(M_prev, Ms[image_id])
    #         print(M_prev.shape)
    #         print(M_prev)
    #         curr_image = get_stitched_image(curr_image, prev_image, M)

    print('show result')
    cv2.imshow('prev_image', prev_image)
    cv2.imshow('prev_label', prev_label)
    cv2.waitKey()


# Call main function
if __name__=='__main__':
    main1()