import cv2
import numpy as np
import os

def clean_ironman(gray):
    # Binary threshold for white drawing/noise on black background
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Connected components: keep only components above a small area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    clean = np.zeros_like(bw)

    min_area = 8   # try 4, 5, 6, 8 and keep the best result
    for i in range(1, num_labels):   # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean

def clean_scenery(img):
    # Moderate denoising: visible improvement without excessive blur
    nlm = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        27,   # h
        27,   # hColor
        7,    # templateWindowSize
        21    # searchWindowSize
    )
    clean = cv2.bilateralFilter(nlm, 7, 50, 50)
    return clean


def process_image(path):
    name = os.path.basename(path).lower()

    if "iron" in name:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read {path}")
        clean = clean_ironman(gray)
        out_name = path.replace(".jpg", "_clean.png")
        cv2.imwrite(out_name, clean)
        cv2.imshow("Original", gray)
        cv2.imshow("Cleaned", clean)

    else:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
        clean = clean_scenery(img)
        out_name = path.replace(".jpg", "_clean.png")
        cv2.imwrite(out_name, clean)
        cv2.imshow("Original", img)
        cv2.imshow("Cleaned", clean)

    print(f"Saved: {out_name}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    images = ["iron_man_noisy.jpg", "noisy.jpg"]
    for img_path in images:
        process_image(img_path)