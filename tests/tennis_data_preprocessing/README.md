# 🎾 Data Cleaning Procedure for Camera-Based Tennis Ball Detection

This tool helps clean and refine the detections of tennis balls from multi-camera setups. The goal is to ensure that only accurate tennis ball positions are retained from the original detections.

---

## 1. Camera Detection Cleaning

We aim to filter out incorrect detections and retain only valid tennis ball points from each camera.

### 🔧 Running the Data Cleaning App

A simple GUI tool has been developed for interactive cleaning.

Run the script:
```bash
python tests/tennis_data_preprocessing/DataCleaner.py
```

### 🧰 Features

#### 📂 Load Bag Files
- Press `o` to open and read detections from ROS bag files.
- Modify the global variable `INITIALDIR` to set the directory containing the bag files.

#### 🏓 Trajectory Labeling
- Use `Right` / `Left` arrow keys to increase or decrease the number of detections in the current trajectory.
- Use `Ctrl + Right` / `Ctrl + Left` for faster adjustments.
- Press `Up` / `Down` arrows to move to the next or previous trajectory.

#### 🧹 Anomaly Removal
- **Single left-click** to select a single point.
- **Click and drag** to select multiple points at once.
- **Right-click** to deselect all selected points.
- Press `d` to delete all selected points (⚠️ irreversible).

#### 💾 Saving Cleaned Data
- Set the global variable `SAVE_DIR` to specify where to save the results.
- Press `s` to save the cleaned data.
- Output files are saved as `.json` with the same name as the input `.bag`, e.g., `example.bag` → `example.json`.

### 📁 Output Format
The resulting `.json` file will contain data in the format:
```json
{
  "camera_<id>": [
    [trajectory_index, 0, seconds, 0, u, v],
    ...
  ]
}
```
## 2. Triangulated Data Cleanning
Format of saved points:
```python
 [trajectory_idx, timestamp, x, y, z, 0, 0, 0, 1, 0, 0]
```
Note: the last 6 values are placeholders for velocity (v) and angular velocity (w)
