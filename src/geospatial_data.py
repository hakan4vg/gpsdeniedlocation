import osmnx as ox
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple
import geopandas as gpd
from shapely.geometry import box
import math

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
        
        min_x, min_y, max_x, max_y = gdf.total_bounds
        
        image_width = 500
        image_height = 500
        
        for _, row in gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                x, y = row.geometry.x, row.geometry.y
                
                px = int((x - min_x) / (max_x - min_x) * image_width)
                py = int((y - min_y) / (max_y - min_y) * image_height)
                
                osm_image = np.zeros((image_height, image_width), dtype=np.uint8)
                
                cv2.circle(osm_image, (px, py), 3, 255, -1)
                
                keypoints, descriptors = self.orb.detectAndCompute(osm_image, None)
                
                if descriptors is not None and len(descriptors) > 0:
                    osm_features.append({
                        'geometry': row.geometry,
                        'descriptor': descriptors[0]
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
                
                keypoints, descriptors = self.orb.detectAndCompute(osm_image, None)
                
                if descriptors is not None and len(descriptors) > 0:
                    osm_features.append({
                        'geometry': row.geometry.centroid,
                        'descriptor': descriptors[0]
                    })
        return osm_features

    def match_features(self, image_features: Tuple[list, np.ndarray]):
        """Match image descriptors to OSM feature descriptors."""
        keypoints, descriptors = image_features
        if descriptors is None:
            return [], []
        
        # Match descriptors with  BFMatcher
        osm_descriptors = np.array([f['descriptor'] for f in self.osm_features], dtype=np.uint8)
        print(f"Shape of image descriptors: {descriptors.shape}")
        print(f"Shape of osm descriptors: {osm_descriptors.shape}")

        matches = self.bf.match(descriptors, osm_descriptors)

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]  # Top 50 matches
        
        matched_locations = [self.osm_features[m.trainIdx]['geometry'] for m in good_matches]
        matched_keypoints = [keypoints[m.queryIdx] for m in good_matches]
        
        return matched_keypoints, matched_locations
