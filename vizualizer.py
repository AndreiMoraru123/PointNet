import plotly.graph_objects as go
import numpy as np


def visualize_rotate(data):

    x_eye = 1.25
    y_eye = 1.25
    z_eye = 0.6
    frames = []

    for i in range(0, 360, 2):
        frames.append(go.Frame(data=data,
                               layout=go.Layout(scene_camera_eye=dict(x=x_eye * np.cos(i * np.pi / 180),
                                                                      y=y_eye * np.sin(i * np.pi / 180),
                                                                      z=z_eye))))
    fig = go.Figure(data=data,
                    layout=go.Layout(
                        updatemenus=[dict(type='buttons',
                                          showactive=False,
                                          y=1,
                                          x=0.8,
                                          xanchor='left',
                                          yanchor='bottom',
                                          pad=dict(t=45, r=10),
                                          buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=50, redraw=True),
                                                                         transition=dict(duration=0),
                                                                         fromcurrent=True,
                                                                         mode='immediate'
                                                                         )]
                                                        )
                                                   ]
                                          )
                                     ]
                    ),
                    frames=frames
                    )

    fig.update_layout(title_text="Object Representation")
    fig.update_traces(marker=dict(size=2, line=dict(width=2, color="DarkSlateGrey")), selector=dict(mode="markers"))
    return fig


def pcshow(xs, ys, zs):
    data = [go.Scatter3d(x=xs, y=ys, z=zs, mode="markers")]
    fig = visualize_rotate(data)
    fig.update_layout(title_text="3D Point Cloud")
    fig.show()
