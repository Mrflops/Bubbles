import cv2
import pupil_apriltags as apriltag
import numpy as np

# Known parameters
KNOWN_TAG_SIZE = 0.165  # in meters, change this to your tag's actual size
FOCAL_LENGTH = 700  # in pixels, you need to determine this for your camera

def draw_tag(image, corners, tag_id, distance):
    # Draw lines between corners
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    # Draw tag ID and distance
    cv2.putText(image, f"ID: {tag_id}", tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(image, f"Dist: {distance:.2f}m", (corners[0][0], corners[0][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def estimate_distance(corners):
    # Calculate the perceived widths between opposite corners
    width1 = np.linalg.norm(corners[0] - corners[1])
    width2 = np.linalg.norm(corners[1] - corners[2])
    width3 = np.linalg.norm(corners[2] - corners[3])
    width4 = np.linalg.norm(corners[3] - corners[0])
    # Average the perceived widths
    average_width = (width1 + width2 + width3 + width4) / 4.0
    # Estimate the distance to the tag
    distance = (KNOWN_TAG_SIZE * FOCAL_LENGTH) / average_width
    return distance, (width1, width2, width3, width4), average_width

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create an AprilTag detector
    detector = apriltag.Detector()
    # Detect tags in the image
    tags = detector.detect(gray)

    for tag in tags:
        corners = tag.corners.astype(int)
        tag_id = tag.tag_id
        distance, perceived_widths, average_width = estimate_distance(corners)

        # Debugging output
        print(f"Tag ID: {tag_id}")
        print(f"Corners: {corners}")
        print(f"Perceived Widths: {perceived_widths}")
        print(f"Average Perceived Width: {average_width}")
        print(f"Estimated Distance: {distance:.2f}m")

        draw_tag(image, corners, tag_id, distance)

    # Save the output image
    output_image_path = 'output_image.png'
    cv2.imwrite(output_image_path, image)
    print(f"Output image saved to {output_image_path}")

if __name__ == '__main__':
    image_path = 'apriltagrobots_overlay.jpg'  # Replace with the path to your image
    main(image_path)
