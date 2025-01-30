import matplotlib.pyplot as plt
import cv2
import config as config
import drone_localization as drone_localization
import image_processing as image_processing
import geospatial_data as geospatial_data

def plot_matches(image, keypoints, osm_points):
    """Visualize matched keypoints and OSM points."""
    plt.figure(figsize=(12, 6))
    
    # Plot drone image with keypoints
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter([kp.pt[0] for kp in keypoints], [kp.pt[1] for kp in keypoints], c='r', s=10)
    plt.title('Drone Image Keypoints')
    
    # Plot OSM points
    plt.subplot(122)
    plt.scatter([p.x for p in osm_points], [p.y for p in osm_points], c='b', s=10)
    plt.title('Matched OSM Landmarks')
    plt.show()
    plt.savefig("data/matches.png")

def main():
    cfg = config.load_config()
    locator = drone_localization.DroneLocator(cfg)
    processor = image_processing.ImageProcessor(cfg)
    geodata = geospatial_data.GeospatialData(cfg)
    
    # Synthetic test image (replace with real data in the future)
    test_image_path = cfg['test_image_path'] if 'test_image_path' in cfg else "synthetic_test_image.png"
    test_image = cv2.imread(test_image_path)  # Generate using OSMnx
    features = processor.extract_features(test_image)
    print(f"Feature Extraction Output:")
    print(f"  Keypoints type: {type(features[0]) if features[0] else None}, Length: {len(features[0]) if features[0] else 0}")
    print(f"  Descriptors type: {type(features[1]) if features[1] is not None else None}, Shape: {features[1].shape if features[1] is not None else None}")

    location_data = geodata.match_features(features)
    print(f"\nFeature Matching Output:")
    print(f"  Matched Keypoints type: {type(location_data[0])}, Length: {len(location_data[0])}")
    print(f"  Matched Locations type: {type(location_data[1])}, Length: {len(location_data[1])}")

    
    plot_matches(test_image, location_data[0], location_data[1])
    
    # Localization
    drone_pose = locator.localize_drone(features, location_data)
    print(f"Estimated Pose: {drone_pose}")

if __name__ == "__main__":
    main()
