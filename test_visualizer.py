from Transforms import PointSampler, RandomRotation, Normalize, PointCloudToTensor, RandomScale, RandomTranslation, RandomNoise
from visualizer import pcshow, visualize_rotate
import plotly.graph_objects as go
import numpy as np

# This will open many tabs in your browser

with open(r"D:\PointNet\ModelNet10\sofa\train\sofa_0011.off", 'r') as f:
    f.readline().strip()

    n_verts, n_faces, _ = tuple([int(s) for s in f.readline().strip().split(' ')])

    verts = [[float(s) for s in f.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in f.readline().strip().split(' ')][1:] for i_face in range(n_faces)]


i, j, k = np.array(faces).T
x, y, z = np.array(verts).T

fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightpink', opacity=0.50)])
fig.update_layout(title_text="Simple 3D Scatter Plot")
visualize_rotate(fig.data).show()

pcshow(*PointSampler(3000)((verts, faces)).T)

# random rotation
pcshow(*RandomRotation()(PointSampler(3000)((verts, faces))).T)

# random scale
pcshow(*RandomScale()(PointSampler(3000)((verts, faces))).T)

# random translation
pcshow(*RandomTranslation()(PointSampler(3000)((verts, faces))).T)

# random noise
pcshow(*RandomNoise()(PointSampler(3000)((verts, faces))).T)
