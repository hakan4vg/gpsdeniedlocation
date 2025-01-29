# GPS-Denied Drone Localization via Visual-Geospatial Cross-Referencing

## Project Overview

This project aims to enable drones to autonomously determine their 3DOF position (latitude, longitude, altitude) in GPS-denied environments. It leverages real-time camera imagery and open-source geospatial data (e.g., satellite maps, OpenStreetMap) to achieve robust and efficient localization.

## Roadmap

Refer to `roadmap.md` for the detailed project roadmap and phased implementation plan.

## Technical Stack

- Python, C++, OpenCV, OSMnx, TensorFlow Lite, GDAL, ROS 2

## Project Structure

```
gpsdeniedlocation/
├── roadmap.md
├── README.md
├── src/
│   ├── main.py
│   ├── drone_localization.py
│   ├── image_processing.py
│   ├── geospatial_data.py
│   └── config.py
├── data/
├── models/
├── scripts/
├── requirements.txt
└── ...
```

## Getting Started

1.  **Installation**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configuration**:
    - Modify `src/config.py` to set project parameters.
3.  **Running**:
    ```bash
    python src/main.py
    ```

## License

[MIT License]

---
