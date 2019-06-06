import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import nibabel as nib
from skimage import measure


def show_image(path):
    fig, ax = plt.subplots(figsize=(8, 8))
    image = mpimg.imread(path)
    ax.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()


def enable_plotly_in_cell():
    # https://stackoverflow.com/a/54771665
    import IPython
    from plotly.offline import init_notebook_mode
    display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    init_notebook_mode(connected=False)


def get_3D_volume(path, opacity=1, color="red"):
    #enable_plotly_in_cell()
    data = nib.load(path).get_data()
    verts, faces = measure.marching_cubes_classic(data, level=0.)
    x,y,z=zip(*verts)
    i,j,k=zip(*faces)

    lighting_effects = dict(ambient=0.6, diffuse=.5, roughness = 0.2, specular=.5, fresnel=0.2)
    trace = go.Mesh3d(x=x,y=y,z=z,i=i,j=j,k=k,opacity=opacity, color=color, flatshading=None, 
                      lighting=lighting_effects, lightposition=dict(x=25,y=100,z=50))
    #alphahull=1, intensity=values
    #plotly.offline.iplot([trace])
    return trace


def plot_3D_volume(path,):
    enable_plotly_in_cell()
    bundle = get_3D_volume(path, color="salmon")
    plotly.offline.iplot([bundle])


def plot_3D_volume_with_brain_mask(path, brain_mask_path):
    enable_plotly_in_cell()
    brain_mask = get_3D_volume(brain_mask_path, opacity=0.1, color="silver")
    bundle = get_3D_volume(path, color="salmon")
    plotly.offline.iplot([brain_mask, bundle])
  