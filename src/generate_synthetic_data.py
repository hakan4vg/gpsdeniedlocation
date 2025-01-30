import osmnx as ox
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # This is required i dont know why
import matplotlib.pyplot as plt

def generate_synthetic_image():
    # Fetch OSM data
    gdf = ox.features_from_point((47.3769, 8.5417), tags={'building': True}, dist=100)
    
    
    dpi = 100
    width = 8
    height = 8
    width_px = int(width * dpi)  # width_px = 800
    height_px = int(height * dpi)  # height_px = 800
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    
    
    ox.plot_footprints(gdf, ax=ax, show=False, close=True)
    fig.canvas.draw()
    
    
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    print(f"Shape of img before reshape: {img.shape}")
    width_px = int(width * dpi)
    height_px = int(height * dpi)
    img = img.reshape(height_px, width_px, 4)[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("data/synthetic_test_image.png", img)

if __name__ == "__main__":
    generate_synthetic_image()
