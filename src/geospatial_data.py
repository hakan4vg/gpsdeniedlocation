import osmnx as ox
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Tuple

class GeospatialData:
    def __init__(self, config):
        self.config = config
        self.landmark_types = config['geospatial_data']['landmark_types']
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Precompute OSM features with synthetic descriptors
        self.osm_features = self._load_osm_features()

    def _load_osm_features(self):
        """Load OSM data and generate synthetic descriptors for landmarks."""
        lat, lon = 47.3769, 8.5417  # Example: Zurich
        tags = {tag: True for tag in self.landmark_types}
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=500)
        
        # Generate synthetic keypoints/descriptors for OSM features
        synthetic_descriptors = []
        for _, row in gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                # dummy descriptor. TODO will change
                desc = np.random.randint(0, 256, (32,), dtype=np.uint8)
                synthetic_descriptors.append({
                    'geometry': row.geometry,
                    'descriptor': desc
                })
            elif row.geometry.geom_type == 'Polygon':
                centroid = row.geometry.centroid
                desc = np.random.randint(0, 256, (32,), dtype=np.uint8)
                synthetic_descriptors.append({
                    'geometry': centroid,
                    'descriptor': desc
                })
        return synthetic_descriptors

    def match_features(self, image_features: Tuple[list, np.ndarray]):
        """Match image descriptors to OSM feature descriptors."""
        keypoints, descriptors = image_features
        if descriptors is None:
            return [], []
        
        # Match descriptors with  BFMatcher
        osm_descriptors = np.array([f['descriptor'] for f in self.osm_features])
        matches = self.bf.match(descriptors, osm_descriptors)
        
        # Filter matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]  # Top 50 matches
        
        matched_locations = [self.osm_features[m.trainIdx]['geometry'] for m in good_matches]
        matched_keypoints = [keypoints[m.queryIdx] for m in good_matches]
        
        return matched_keypoints, matched_locations
