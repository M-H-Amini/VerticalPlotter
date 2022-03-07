import cv2
import numpy as np
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def kmeans_seg_gray(image, k):
    ##  Clustering...
    kmeans = KMeans(n_clusters=k, random_state=0).fit(np.reshape(image, (-1, 1)))
    centers = kmeans.cluster_centers_
    mask = np.reshape(kmeans.labels_, image.shape)

    ##  Reconstructing...
    segmented_image = mask.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            segmented_image[i, j] = centers[mask[i, j], 0]
    segmented_image = segmented_image.astype(np.uint8)

    return segmented_image, centers, mask


def _contourize(image, thresh_val):
    _, thresh = cv2.threshold(image, thresh_val, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _contourizeAdaptive(image):
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contourize(
    image, thresh_ranges, contours_percentage, image_path="cnt.png", plot=False
):
    contours = []
    for i in thresh_ranges:
        contours.extend(_contourize(image, i))
    max_area_index = int(len(contours) * contours_percentage)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-max_area_index:]
    result = 255.0 * np.ones_like(image)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    if plot:
        cv2.imshow("Contourize", result)
        cv2.waitKey()
        cv2.destroyAllWindows()

    pickle_name = image_path[:-3] + "pickle"
    pickle_out = open(pickle_name, "wb")
    pickle.dump(contours, pickle_out)
    pickle_out.close()
    return result, contours


def adaptiveContourize(image, contours_percentage, plot=False):
    contours = []
    for i in thresh_ranges:
        contours.extend(_contourizeAdaptive(image))
    max_area_index = int(len(contours) * contours_percentage)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[-max_area_index:]
    if plot:
        result = 255.0 * np.ones_like(image)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
        cv2.imshow("Adaptive Contourize", result)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return contours


def animateContours(image, contours):
    plane = np.ones_like(image) * 255
    for i in range(len(contours)):
        visited = contours[i]
        visited = np.squeeze(visited, 1)
        visited = visited.tolist()
        for j in range(len(visited)):
            plane[visited[j][1], visited[j][0]] = 0
        cv2.imshow("Contours", plane)
        cv2.waitKey()


if __name__ == "__main__":
    image_path = "S3.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (800, 600))
    cv2.imwrite("gray.jpg", image)
    _, thresh = cv2.threshold(image, 70, 255, 0)
    cv2.imwrite("thresh.jpg", thresh)
    image_denoised = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite("denoised.jpg", image_denoised)
    print("Segmenting image...")
    image_segmented, _, _ = kmeans_seg_gray(image, 4)
    cv2.imwrite("segmented.jpg", image_segmented)
    print("Segmentation done...")
    cv2.imshow("b", image)
    cv2.waitKey()
    thresh_ranges = [20, 40, 60, 90]
    max_area_index = 1
    result, contours = contourize(
        image, thresh_ranges, max_area_index, image_path, True
    )
    # contours = adaptiveContourize(image, max_area_index, True)
    print("len...", len(contours))
    # animateContours(image, contours)
    cv2.imwrite("contours.jpg", result)
    cv2.imshow("result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    pickle_name = image_path[:-3] + "pickle"
    pickle_out = open(pickle_name, "wb")
    pickle.dump(contours, pickle_out)
    pickle_out.close()
