# From https://github.com/ryota-skating/FS-Jump3D/blob/main/utils/visualization.ipynb

import os
import numpy as np
import plotly.graph_objects as go

H36M_CONNECTIONS = [
    (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
    (12, 13), (14, 15), (15, 16)
    ]
FSJUMP3D_CONNECTIONS = [
    (0, 3), (0, 45), (1, 4), (1, 22), (2, 5), (2, 23), (3, 6), (3, 24),
    (4, 6), (4, 7), (4, 8), (4, 83), (5, 6), (5, 7), (5, 83), (6, 8),
    (7, 8), (8, 9), (9, 10), (9, 11), (9, 14), (10, 12), (11, 13),
    (12, 13), (14, 15), (15, 16), (15, 17), (15, 18), (15, 19), 
    (16, 20), (17, 21), (22, 23), (22, 24), (22, 25), (23, 24),
    (23, 25), (24, 26), (25, 26), (26, 27), (27, 28), (27, 29),
    (27, 32), (28, 30), (29, 31), (30, 31), (32, 33), (33, 34),
    (33, 35), (33, 36), (33, 37), (34, 38), (35, 39), (40, 42),
    (40, 43), (41, 42), (41, 43), (44, 55), (44, 70), (45, 47),
    (45, 54), (45, 69), (46, 56), (46, 71), (47, 52), (48, 50),
    (48, 51), (49, 50), (49, 51), (53, 55), (53, 56), (53, 57),
    (53, 58), (54, 55), (54, 56), (56, 71), (58, 60), (58, 61),
    (58, 62), (59, 60), (59, 61), (62, 63), (62, 64), (62, 65),
    (63, 66), (64, 67), (68, 70), (68, 71), (68, 72), (69, 70),
    (69, 71), (72, 73), (73, 75), (73, 76), (73, 77), (74, 75),
    (74, 76), (77, 78), (77, 79), (77, 80), (78, 81), (79, 82)
    ]

def show_3D_pose_plotly(pose3d: np.ndarray, connections: list):
    '''Show 3D pose using Plotly.'''

    # Scatter points and keypoint indices
    scatter_points = go.Scatter3d(
        x=pose3d[:, 0],
        y=pose3d[:, 1],
        z=pose3d[:, 2],
        mode='markers+text',
        marker=dict(size=1.5, color='black'),
        text=[str(i) for i in range(pose3d.shape[0])],
        textposition="top center"
    )

    # Create lines
    lines = []
    for c1, c2 in connections:
        x, y, z = [np.array([pose3d[c1, j], pose3d[c2, j]]) for j in [0, 1, 2]]
        lines.append(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(width=4, color='red'),
            showlegend=False
        ))

    # Combine scatter points and lines
    fig = go.Figure(data=[scatter_points] + lines)

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[pose3d[0, 0] - 1000, pose3d[0, 0] + 1000]),
            yaxis=dict(title='Y', range=[pose3d[0, 1] - 1000, pose3d[0, 1] + 1000]),
            zaxis=dict(title='Z', range=[pose3d[0, 2] - 1000, pose3d[0, 2] + 1000]),
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False
        ),
        scene_aspectmode='cube',
        margin=dict(r=10, l=10, b=10, t=10),
        width=800, height=800
    )

    fig.show()

# def main():
#     # Load 3D pose data
#     file_path = 'CVPR2024-FACT/data/fsjump/features/Axel_1.npy'  # Set your path to 3D pose npy file.
#     pose3d = np.load(file_path)

#     frame_idx = 0  # Set the frame index to visualize.
#     connections = H36M_CONNECTIONS  # Set keypoint connections rig.

#     ## Visualize 3D pose
#     # show_3D_pose_plotly(pose3d[frame_idx], connections)
#     # Fix for flat array
#     if pose3d.ndim == 1:
#         pose3d = pose3d.reshape(-1, 3)
#     show_3D_pose_plotly(pose3d, connections)

def main():
    # Load 3D pose data
    file_path = 'CVPR2024-FACT/data/fsjump/features/Lutz_1.npy'
    pose3d = np.load(file_path)

    # Print shape to debug
    print("Loaded pose3d shape:", pose3d.shape)

    frame_idx = 200
    connections = H36M_CONNECTIONS # or FSJUMP3D_CONNECTIONS if you're using a full-body model

    # Reshape to (T, 17, 3)
    pose3d = pose3d.reshape(pose3d.shape[0], -1, 3)
    print("Reshaped pose3d shape:", pose3d.shape)  # (T, 17, 3)

    # Visualize the selected frame
    show_3D_pose_plotly(pose3d[frame_idx], connections)

if __name__ == '__main__':
    main()