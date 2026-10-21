"""
Utility script to generate the ArUco marker for My Robot.
Saves a high-resolution PNG image with a border and heading orientation marker.
"""

import cv2
import numpy as np

def generate_my_bot_marker(marker_id: int = 0, size_px: int = 500, output_file: str = "my_bot_aruco_tag.png"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Check if modern generateImageMarker exists (OpenCV >= 4.7.0)
    if hasattr(cv2.aruco, 'generateImageMarker'):
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    else:
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, size_px)
        
    # Add a white border around the tag for high contrast on robot top plate
    border = 60
    bordered = cv2.copyMakeBorder(
        marker_img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255
    )
    
    # Draw a clear FRONT direction arrow on the top margin
    h, w = bordered.shape
    cv2.arrowedLine(bordered, (w // 2, border - 15), (w // 2, 15), 0, 3, tipLength=0.4)
    cv2.putText(bordered, "FRONT", (w // 2 - 30, border - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    cv2.imwrite(output_file, bordered)
    print(f"[OK] Generated ArUco Tag (ID: {marker_id}, DICT_4X4_50) -> {output_file}")
    print("     Mount this tag flat on top armor with the FRONT arrow facing the robot's front bumper.")

if __name__ == "__main__":
    generate_my_bot_marker()
