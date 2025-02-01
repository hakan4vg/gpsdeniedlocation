import matplotlib.pyplot as plt
import cv2
import config as config
import drone_localization as drone_localization
import image_processing as image_processing
import geospatial_data as geospatial_data
import generate_synthetic_data
import osmnx as ox

def plot_matches(image, keypoints, osm_points):
    """Visualize matched keypoints and OSM points."""
    plt.figure(figsize=(12, 6))

    # Plot drone image with keypoints
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter([kp.pt[0] for kp in keypoints], [kp.pt[1] for kp in keypoints], c='r', s=10)
    plt.title("Drone Image Keypoints")

    # Plot OSM points
    plt.subplot(122)
    if osm_points:
        plt.scatter([p.x for p in osm_points], [p.y for p in osm_points], c='b', s=10)
    plt.title("OSM Features")

    # --- Add these lines to adjust plot limits ---
    if keypoints and osm_points:
        all_x_coords_kp = [kp.pt[0] for kp in keypoints]
        all_y_coords_kp = [kp.pt[1] for kp in keypoints]
        all_x_coords_osm = [p.x for p in osm_points]
        all_y_coords_osm = [p.y for p in osm_points]

        all_x_coords = all_x_coords_kp + all_x_coords_osm
        all_y_coords = all_y_coords_kp + all_y_coords_osm

        if all_x_coords and all_y_coords: 
            plt.subplot(121) 
            min_x, max_x = min(all_x_coords), max(all_x_coords)
            plt.gca().invert_yaxis() 

            plt.subplot(122) # 
            min_y, max_y = min(all_y_coords), max(all_y_coords)
            plt.gca().invert_yaxis()

    plt.savefig("data/matches.png")

def main():
    cfg = config.load_config()
    locator = drone_localization.DroneLocator(cfg)
    processor = image_processing.ImageProcessor(cfg)
    geodata = geospatial_data.GeospatialData(cfg)
    
    # Generate synthetic image
    generate_synthetic_data.generate_synthetic_image()
    
    test_image = processor.get_image()
    features = processor.extract_features(test_image)
    if features[1] is not None:
        print(f"Image Descriptors - Shape: {features[1].shape}, Dtype: {features[1].dtype}")
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
