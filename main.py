import os
import platform

if platform.system() == "Linux":
    os.environ["WEBOTS_HOME"] = "/usr/local/webots"

import numpy as np
import math
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
from skimage.draw import line

import networkx as nx

from Ransac import Ransac


def find_all_cycles(graph):
    def dfs_cycle(start, current, visited, stack, cycles):
        visited[current] = True
        stack.append(current)

        for neighbor in graph.neighbors(current):
            if not visited[neighbor]:
                dfs_cycle(start, neighbor, visited, stack, cycles)
            elif neighbor == start and len(stack) > 2:
                cycle = stack[:] + [start]
                cycles.append(cycle)

        stack.pop()
        visited[current] = False

    cycles = []
    for node in graph.nodes():
        visited = {n: False for n in graph.nodes()}
        dfs_cycle(node, node, visited, [], cycles)

    # Remove duplicate cycles (considering rotations and reversed versions)
    unique_cycles = []
    for cycle in cycles:
        cycle = cycle[:-1]  # Remove the duplicate start/end node
        normalized_cycle = tuple(sorted(cycle))
        if normalized_cycle not in unique_cycles:
            unique_cycles.append(normalized_cycle)

    return [list(cycle) for cycle in unique_cycles]



def find_cycles(edges):
    # Create the graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Find all cycles in the graph
    cycles = find_all_cycles(G)

    return cycles


angles_dict = {
    'draw_triangle': 60,
    'draw_square': 90,
    'draw_pentagon': 36,
    'draw_plane': 0,
    'draw_unknown': 0
}


def sort_vertices_by_angle(vertices, centroid):
    centroid_x, centroid_y = centroid
    sorted_vertices = sorted(vertices, key=lambda v: np.arctan2(v[1] - centroid_y, v[0] - centroid_x))
    return sorted_vertices


def is_cycle(vertices):
    G = nx.Graph()
    num_vertices = len(vertices)
    for i in range(num_vertices):
        G.add_edge(tuple(vertices[i]), tuple(vertices[(i + 1) % num_vertices]))
    if len(list(nx.cycle_basis(G))) == 1 and all(len(list(G.neighbors(node))) == 2 for node in G.nodes):
        return True
    return False


def classify_shape(vertices, centroid):
    num_vertices = len(vertices)

    sorted_vertices = sort_vertices_by_angle(vertices, centroid)

    if num_vertices == 2:
        return "Plane"

    if not is_cycle(vertices):
        return f"Unknown with {num_vertices} vertices"

    if num_vertices == 3:
        side_lengths = []
        for i in range(num_vertices):
            x1, y1 = sorted_vertices[i]
            x2, y2 = sorted_vertices[(i + 1) % num_vertices]
            side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            side_lengths.append(side_length)
        if all(abs(length - side_lengths[0]) < 0.1 for length in side_lengths):
            return "Regular Triangle"
        else:
            return "Triangle"

    elif num_vertices == 4:
        side_lengths = []
        for i in range(num_vertices):
            x1, y1 = sorted_vertices[i]
            x2, y2 = sorted_vertices[(i + 1) % num_vertices]
            side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            side_lengths.append(side_length)

        # Calculate diagonals
        diagonal1 = math.sqrt(
            (sorted_vertices[0][0] - sorted_vertices[2][0]) ** 2 + (sorted_vertices[0][1] - sorted_vertices[2][1]) ** 2)
        diagonal2 = math.sqrt(
            (sorted_vertices[1][0] - sorted_vertices[3][0]) ** 2 + (sorted_vertices[1][1] - sorted_vertices[3][1]) ** 2)

        if all(abs(length - side_lengths[0]) < 0.1 for length in side_lengths) and abs(diagonal1 - diagonal2) < 0.1:
            return "Square"
        elif (abs(side_lengths[0] - side_lengths[2]) < 0.1 and
              abs(side_lengths[1] - side_lengths[3]) < 0.1 and
              abs(diagonal1 - diagonal2) < 0.1):
            return "Rectangle"
        else:
            return "Polygon with 4 vertices"

    elif num_vertices == 5:
        side_lengths = []
        for i in range(num_vertices):
            x1, y1 = sorted_vertices[i]
            x2, y2 = sorted_vertices[(i + 1) % num_vertices]
            side_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            side_lengths.append(side_length)
        if all(abs(length - side_lengths[0]) < 0.1 for length in side_lengths):
            return "Pentagon"
        else:
            return "Polygon with 5 vertices"

    else:
        return f"Polygon with {num_vertices} vertices"

    threshold = 0.02
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

def calculate_rotation_angle(rotated_vertices, type_shape):
    # Sort vertices based on their y-coordinate
    sorted_vertices = sorted(rotated_vertices, key=lambda vertex: vertex[1])

    # Get the first two points (lowest y-coordinate)
    p1, p2 = sorted_vertices[:2]

    # Calculate the angle of the line connecting p1 and p2 with respect to the x-axis
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    angle_degrees = np.degrees(angle)
    # Convert angle to degrees and adjust according to shape
    rotation_angle = angle_degrees % angles_dict[type_shape]

    return rotation_angle


def parse_vertices(vertices_str):
    vertices = vertices_str.replace("(", "").replace(")", "").split(";")
    return [tuple(map(int, v.split(","))) for v in vertices]



def main() -> None:
    mapname = "test_map_3"
    pcd_filename = f"point_clouds/{mapname}.npy"
    output_csv = f"results/comparison_results_{mapname}.csv"
    ground_truths = pd.read_csv(f"ground_truth/{mapname}_shapes.csv")

    data_arr = np.load(pcd_filename)
    data_arr = data_arr[data_arr[:, -1] >= 0]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])

    planes, intersection_points, intersection_edges, inliers_all = Ransac(data_arr,
                                                                          min_points=500,
                                                                          threshold=0.03,
                                                                          max_iteration=50000)

    shapes = find_cycles(intersection_edges)

    selected_points = []
    for cycle in shapes:
        if cycle:
            selected_points.append([intersection_points[idx] for idx in cycle])

    cycle_centers = []
    for cycle in shapes:
        cycle_points = intersection_points[cycle]
        center = np.mean(cycle_points, axis=0)
        cycle_centers.append(center)

    cycle_centers = np.array(cycle_centers)

    orientation_angles = []
    detected_shapes = []

    for points, center in zip(selected_points, cycle_centers):
        flat_points = [[point[0], point[1]] for point in points]
        classification = classify_shape(flat_points, center[:2])
        print(classification)

        shape_key = ''
        if classification == "Triangle" or classification == "Regular Triangle" or classification == "Polygon with 3 vertices":
            shape_key = 'draw_triangle'
        elif classification == "Square" or classification == "Rectangle" or classification == "Polygon with 4 vertices":
            shape_key = 'draw_square'
        elif classification == "Pentagon" or classification == "Regular Pentagon" or classification == "Polygon with 5 vertices":
            shape_key = 'draw_pentagon'
        elif classification == "Plane":
            shape_key = 'draw_plane'
        else:
            shape_key = 'draw_unknown'

        rotation_angle = calculate_rotation_angle(flat_points, shape_key)
        orientation_angles.append(rotation_angle)
        detected_shapes.append(classification)


    # Ensure all arrays have the same length
    num_points = len(intersection_points)
    num_centers = len(cycle_centers)
    num_angles = len(orientation_angles)

    repeated_centers = np.repeat(cycle_centers, num_points // num_centers + 1, axis=0)[:num_points]
    repeated_angles = np.repeat(orientation_angles, num_points // num_angles + 1)[:num_points]

    results_df = pd.DataFrame({
        'intersection_x': intersection_points[:, 0],
        'intersection_y': intersection_points[:, 1],
        'intersection_z': intersection_points[:, 2],
        'center_x': repeated_centers[:, 0],
        'center_y': repeated_centers[:, 1],
        'center_z': repeated_centers[:, 2],
        'angle_rad': repeated_angles,
        'angle_deg': [angle if angle is not None else None for angle in repeated_angles]
    })

    os.makedirs("./results", exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    # Visualize the points and their orientation angles
    plt.figure(figsize=(10, 10))
    for row in ground_truths['Vertices']:
        vertices = np.array(parse_vertices(row)) / 1000

        last_point = vertices[0]
        for point in vertices[1:]:

            plt.plot([last_point[0], point[0]], [last_point[1], point[1]], c='blue', label="Ransac")
            last_point = point
        plt.plot([vertices[0][0], vertices[-1][0]], [vertices[0][1], vertices[-1][1]], c="blue", label="Ground Truth")

    plt.scatter(np.array(inliers_all)[:, 0], np.array(inliers_all)[:, 1], c='orange', label='Inliers')

    for edge in intersection_edges:
        point_a = intersection_points[edge[0]]
        point_b = intersection_points[edge[1]]
        plt.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], c="green", label='Ransac')

    for point, angle in zip(cycle_centers, orientation_angles):
        if angle is not None:
            x, y = point[:2]
            plt.text(x, y, f"{angle:.2f}°", color='red', fontsize=12)

    for center in cycle_centers:
        x, y = center[:2]
        plt.text(x, y - 0.05, f"({x:.2f}, {y:.2f})", color='purple', fontsize=12, ha='center', va='top')

    plt.xlabel('X')
    plt.ylabel('Y')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    plt.title('Source Points, Intersection Points, and Target Points with Orientation Angles and Cycle Centers')
    plt.grid(True)
    plt.show()

    # Visualize the intersection edges and points
    matrix = np.zeros((1030, 1030))
    for edge in intersection_edges:
        point_a = np.round(intersection_points[edge[0]] * 100).astype(int) + 15
        point_b = np.round(intersection_points[edge[1]] * 100).astype(int) + 15

        rr, cc = line(point_a[0], point_a[1], point_b[0], point_b[1])
        matrix[rr, cc] = 1

    for point in intersection_points:
        point = np.round(point * 100).astype(int) + 15
        matrix[point[0], point[1]] = 2

    plt.imshow(np.rot90(matrix))
    plt.show()

    comparison_data = []
    for index, row in ground_truths.iterrows():
        shape = row['Shape']
        vertices = parse_vertices(row['Vertices'])
        gt_angle = row['Angle']
        gt_center = np.array(eval(row['Center']))

        # Find the closest detected shape
        detected_center_idx = np.argmin(np.linalg.norm(cycle_centers[:, :2] * 1000 - gt_center[:2], axis=1))
        detected_center = cycle_centers[detected_center_idx] * 1000
        detected_angle = orientation_angles[detected_center_idx]

        center_error = np.linalg.norm(gt_center[:2] - detected_center[:2])
        angle_error = np.abs(gt_angle - detected_angle) if detected_angle is not None else None

        comparison_data.append({
            'Ground Truth Shape': shape,
            'Detected Shape': detected_shapes[detected_center_idx],
            'Center Error': center_error,
            'Angle Error': angle_error
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df)



if __name__ == '__main__':
    main()
