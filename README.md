 RANSAC-Based Plane Detection: Improving Precision in 3D Mapping

This project is a part of the course "Introdução à Robótica Inteligente" at FCUP. The goal of this project is to improve the precision of 3D mapping by using RANSAC-based plane detection.
The project includes the ```main.py``` file that contains the code for the performance metrics that run the RANSAC algorithm and the plane detection algorithm. It also includes a ```generate_map_matrix.py``` file that generates a map matrix with the ground truth values for the plane detection algorithm and the map for the Webots simulation, a ```scan_pcd.py``` file that generates a point cloud from the Webots simulation, a ```Ransac.py``` file that contains the RANSAC algorithm, and a ```plane.py``` that was taken from pyransac 3d library, but also contains new code written by us.

The ```transformations.py``` and ```utils.py``` files were taken from the code that the teacher provided during the lessons

## How to run the code

This code works in all operating systems. First you need to create a map matrix by running the ```generate_map_matrix.py``` file. Then you need to run the Webots simulation with this different configurations:

#### Robot:
- supervisor: true
- name: "EPUCK"

#### LiDAR:
- number_of_layers: 8
- vertical_fov: 0.4


After this run the ```scan_pcd.py``` file to generate the point cloud. Finally, you can run the ```main.py``` file to run the RANSAC algorithm and the plane detection algorithm.

## Libraries required:
- numpy
- open3d
- scipy
- matplotlib
- pandas
- networkx
- scikit-image
- opencv
- scikit-learn


