import osmnx as ox
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple
import geopandas as gpd
from shapely.geometry import box
import math
import os

class GeospatialData:
    def __init__(self, config):
        self.config = config
        self.landmark_types = config['geospatial_data']['landmark_types']
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create()

        self.mission_center = (47.3769, 8.5417)
        self.mission_radius_km = 2.0
        self.test_subnet_size_m = 50

        self.osm_features = self._load_osm_features()


    def _get_bounding_bbox(self):
        lat, lon = self.mission_center
        delta_lat = self.mission_radius_km / 111.32

        delta_lon = self.mission_radius_km / (111.32 * math.cos(math.radians(lat)))
        
        north = lat + delta_lat
        south = lat - delta_lat
        east = lon + delta_lon
        west = lon - delta_lon
        
        return north, south, east, west


    def _load_osm_features(self):
        """Load OSM data and generate ORB descriptors for landmarks."""
        north, south, east, west = self._get_bounding_bbox()
        tags = {tag: True for tag in self.landmark_types}
        
        bbox = (west, south, east, north)
        gdf = ox.features.features_from_bbox(bbox, tags={'building': True})
        
        try:
            bbox = (west, south, east, north)
            gdf = ox.features.features_from_bbox(bbox, tags={'building': True})
        except Exception as e:
            print(f"Error fetching OSM data: {e}")
            return []
        
        print(f"OSM data fetched successfully. Number of features: {len(gdf)}")        

        osm_features = []
        debug_osm_images_dir = "data/debug_osm_images"
        os.makedirs(debug_osm_images_dir, exist_ok=True)
        
        min_x, min_y, max_x, max_y = gdf.total_bounds
        
        image_width = 500
        image_height = 500
        
        for _, row in gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                x, y = row.geometry.x, row.geometry.y
                
                px = int((x - min_x) / (max_x - min_x) * image_width)
                py = int((y - min_y) / (max_y - min_y) * image_height)
                
                osm_image = np.zeros((image_height, image_width), dtype=np.uint8)
                
                cv2.circle(osm_image, (px, py), 5, 255, -1)
                
                _, descriptors = self.orb.detectAndCompute(osm_image, None)
                cv2.imwrite(os.path.join(debug_osm_images_dir, f"osm_point_{_.index}.png"), osm_image)

                if descriptors is not None and len(descriptors) > 0:
                    osm_features.append({
                        'geometry': row.geometry,
                        'descriptors': descriptors.tolist()
                    })
                
            elif row.geometry.geom_type == 'Polygon':
                
                polygon_coords = np.array(row.geometry.exterior.coords)
                
                polygon_pixels = []
                for x, y in polygon_coords:
                    px = int((x - min_x) / (max_x - min_x) * image_width)
                    py = int((y - min_y) / (max_y - min_y) * image_height)
                    polygon_pixels.append((px, py))
                
                polygon_pixels = np.array(polygon_pixels, dtype=np.int32)
                
                osm_image = np.zeros((image_height, image_width), dtype=np.uint8)
                
                cv2.fillPoly(osm_image, [polygon_pixels], 255)
                
                _, descriptors = self.orb.detectAndCompute(osm_image, None)
                cv2.imwrite(os.path.join(debug_osm_images_dir, f"osm_polygon_{_.index}.png"), osm_image)

                if descriptors is not None and len(descriptors) > 0:
                    osm_features.append({
                        'geometry': row.geometry.centroid,
                        'descriptors': descriptors.tolist()
                    })
        return osm_features

    def match_features(self, image_features: Tuple[list, np.ndarray]):
        """Match image descriptors to OSM feature descriptors."""
        keypoints, descriptors = image_features
        if descriptors is None:
            return [], []
        
        # Match descriptors with  BFMatcher
        osm_descriptors_list = [np.array(f['descriptors'], dtype=np.uint8) for f in self.osm_features if f['descriptors']]
        osm_descriptors_concatenated = np.concatenate(osm_descriptors_list, axis=0) if osm_descriptors_list else None
        print(f"Image Descriptors - Shape: {descriptors.shape if descriptors is not None else None}, Dtype: {descriptors.dtype if descriptors is not None else None}")
        print(f"OSM Descriptors (Concatenated) - Shape: {osm_descriptors_concatenated.shape if osm_descriptors_concatenated is not None else None}, Dtype: {osm_descriptors_concatenated.dtype if osm_descriptors_concatenated is not None else None}")

        if osm_descriptors_concatenated is None or osm_descriptors_concatenated.shape[0] == 0:
            print("No OSM descriptors found")
            return [], []
        
        osm_descriptors = osm_descriptors_concatenated
        print(f"Matching image descriptors against {osm_descriptors.shape[0]} OSM descriptors")
        matches = self.bf.match(descriptors, osm_descriptors)
        

        print(f"Number of matches found: {len(matches)}")
        if matches:
            avg_match_distance = np.mean([m.distance for m in matches])
            print(f"Average match distance: {avg_match_distance}")

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50] if len(matches) > 50 else matches

        matched_osm_feature_indices = [m.trainIdx for m in good_matches]
        matched_locations = []
        osm_descriptor_offset = 0
        for feature in self.osm_features:
            num_descriptors_for_feature = len(feature['descriptors'])
            feature_indices = range(osm_descriptor_offset, osm_descriptor_offset + num_descriptors_for_feature)
            feature_matches = [m_idx for m_idx in matched_osm_feature_indices if m_idx in feature_indices]
            if feature_matches:
                matched_locations.append(feature['geometry'])

            osm_descriptor_offset += num_descriptors_for_feature

        matched_keypoints = [keypoints[m.queryIdx] for m in good_matches]
        
        return matched_keypoints, matched_locations
