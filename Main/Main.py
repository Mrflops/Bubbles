import cv2
import pupil_apriltags as apriltag
import numpy as np

# Known parameters
KNOWN_TAG_SIZE = 0.165  # in meters, change this to your tag's actual size
FOCAL_LENGTH = 700  # in pixels, you need to determine this for your camera

def draw_tag(image, corners, tag_id, distance):
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(image, f"ID: {tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image, f"Dist: {distance:.2f}m", (corners[0][0], corners[0][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def estimate_distance(corners):
    # Calculate the perceived width of the tag in the image
    perceived_width = np.linalg.norm(corners[0] - corners[1])
    # Estimate the distance to the tag
    distance = (KNOWN_TAG_SIZE * FOCAL_LENGTH) / perceived_width
    return distance

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = apriltag.Detector()
    tags = detector.detect(gray)

    for tag in tags:
        corners = tag.corners.astype(int)
        tag_id = tag.tag_id
        distance = estimate_distance(corners)
        draw_tag(image, corners, tag_id, distance)

    cv2.imshow('AprilTag Tracker', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = 'orig_img.jpg'  # Replace with the path to your image
    main(image_path)
