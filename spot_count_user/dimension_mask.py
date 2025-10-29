import numpy as np
import cv2
import os

directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def flood_fill(x, y, label, image, labeled_image):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < image.shape[0] and 0 <= cy < image.shape[1] and image[cx, cy] == 255 and labeled_image[cx, cy] == 0:
            labeled_image[cx, cy] = label

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                stack.append((nx, ny))


def is_within_tolerance(dim, ref_dim, tolerance=0.5):
    lower_bound = ref_dim * (1 - tolerance)
    upper_bound = ref_dim * (1 + tolerance)
    return lower_bound <= dim <= upper_bound


start_point = None
end_point = None
cropping = False


def click_and_crop(event, x, y, flags, param):
    global start_point, end_point, cropping

    # On left mouse button press, record the starting point and set cropping to True
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        cropping = True

    # On mouse movement with the button pressed, update the ending point
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        end_point = (x, y)

    # On left mouse button release, record the ending point and set cropping to False
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        cropping = False


ref_height = 0
ref_width = 0

print("Press 'c' to record dimensions. Press 'q' to close window.")

values = []
path = "spot_plate_images/spot_plates_23-01-25/refined"

# Identify spot by mouseclick
file_path = "spot_plate_images/spot_plates_23-01-25/refined/spot_plate_1.jpg"
image = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
img_or = cv2.imread(file_path)
clone = img_or.copy()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", click_and_crop)

while True:
    temp_image = clone.copy()

    if start_point and end_point:
        cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)

    cv2.imshow("Image", temp_image)
    key = cv2.waitKey(1) & 0xFF

    # Break the loop on pressing 'q'
    if key == ord("q"):
        break

    # On pressing 'c', calculate and print the region dimensions
    elif key == ord("c") and start_point and end_point:
        x1, y1 = start_point
        x2, y2 = end_point

        ref_width = abs(x2 - x1)
        ref_height = abs(y2 - y1)
        print(f"Width: {ref_width} pixels, Height: {ref_height} pixels, Area: {ref_width * ref_height} pixels")

cv2.destroyAllWindows()

for img in os.listdir(path):
    file_path = os.path.join(path, img)
    image = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)

    labeled_image = np.zeros_like(image, dtype=int)
    label_counter = 1
    rows, cols = image.shape

    # 1. Find an initial white pixel guaranteed to be part of the outer region
    outer_x, outer_y = -1, -1
    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255:
                outer_x, outer_y = i, j
                break
        if outer_x != -1:
            break

    if outer_x == -1:
        print(f"No white pixels found in image {file_path}")
        continue

    # 2. Label the outer region with a -1
    labeled_image = np.zeros_like(image, dtype=int)
    flood_fill(outer_x, outer_y, -1, image, labeled_image)

    # 3. Create a color image for visualization
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 4. Spot count
    label_counter = 1
    spot_counts = 0

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255 and labeled_image[i, j] != -1:
                flood_fill(i, j, label_counter, image, labeled_image)
                x, y, w, h = cv2.boundingRect(np.uint8(labeled_image == label_counter))
                if is_within_tolerance(h, ref_height) and is_within_tolerance(w, ref_width):
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    spot_counts += 1
                else:
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                label_counter += 1

    num_spots = spot_counts
    print(f"{img}: {num_spots} spots filtered")
    #values.append(num_spots)

    cv2.imshow(f"{num_spots} Spots Detected", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
