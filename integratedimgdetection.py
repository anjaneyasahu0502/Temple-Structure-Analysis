import cv2
import numpy as np

# 1. Extract meaningful object contours (multiple, not just the largest)
def extract_significant_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive threshold instead of fixed threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 
                                     11, 2)

    # Morphological closing to fill small holes
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    # Find contours and create a full mask from all relevant areas
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Avoid noise
            cv2.drawContours(mask, [cv2.convexHull(contour)], -1, 255, -1)

    # Final object-extracted image
    extracted = cv2.bitwise_and(image, image, mask=mask)
    return extracted


# 2. Feature Matching (Branch & Bound-style) with full image or significant part
def branch_and_bound_feature_matching(object_image, image2):
    gray1 = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("No descriptors found.")
        return None

    matches = []
    for i, d1 in enumerate(des1):
        min_dist = float('inf')
        best_match = None
        for j, d2 in enumerate(des2):
            dist = np.sum(np.bitwise_xor(d1, d2))
            if dist < min_dist:
                min_dist = dist
                best_match = cv2.DMatch(i, j, 0, float(min_dist))
        if best_match:
            matches.append(best_match)

    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(object_image, kp1, image2, kp2, matches[:30], None, flags=2)
    return matched_img

# 3. Backtracking-style segmentation (Otsu + error minimization)
def backtracking_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_mask = None
    min_error = float('inf')
    for t in range(0, 256, 10):
        _, mask = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
        error = np.sum(np.abs(mask - otsu))
        if error < min_error:
            min_error = error
            best_mask = mask

    inverted_mask = cv2.bitwise_not(best_mask)
    segmented = cv2.bitwise_and(image, image, mask=best_mask)
    background = np.zeros_like(image)
    final_output = np.where(inverted_mask[:, :, None] == 255, background, segmented)

    return final_output

# 4. Full Pipeline Runner
def run_integrated_pipeline():
    img1 = cv2.imread("temple1.png")
    img2 = cv2.imread("temple2.png")

    # Step 1: Extract meaningful regions, not just biggest
    object_img = extract_significant_objects(img1)

    # Step 2: Match features
    matched_img = branch_and_bound_feature_matching(object_img, img2)
    if matched_img is None:
        print("Feature matching failed.")
        return

    # Step 3: Segment matched result
    final_output = backtracking_segmentation(matched_img)

    # Show final output
    cv2.imshow("Final Output", final_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
run_integrated_pipeline()