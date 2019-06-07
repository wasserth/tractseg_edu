import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import nibabel as nib
from skimage import measure
from nibabel import trackvis
from dipy.tracking.utils import move_streamlines

def enable_plotly_in_cell():
    # https://stackoverflow.com/a/54771665
    import IPython
    from plotly.offline import init_notebook_mode
    display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
    init_notebook_mode(connected=False)


def get_voxelwise_orientation_colormap(streamlines, orientation="axial"):
    color = []
    for i in streamlines:
        local_color = np.gradient(i, axis=0)
        local_color = np.abs(local_color)
        local_color = (local_color.T / np.max(local_color, axis=1)).T

        local_color_reordered = np.zeros(local_color.shape)
        if orientation == "axial":
            local_color_reordered[:, 0] = local_color[:, 0]
            local_color_reordered[:, 1] = local_color[:, 2]
            local_color_reordered[:, 2] = local_color[:, 1]
        elif orientation == "sagittal":
            local_color_reordered[:, 0] = local_color[:, 2]
            local_color_reordered[:, 1] = local_color[:, 0]
            local_color_reordered[:, 2] = local_color[:, 1]
        else:
            local_color_reordered = local_color
        color.append(local_color_reordered)
    return color


def show_image(path):
    fig, ax = plt.subplots(figsize=(8, 8))
    image = mpimg.imread(path)
    ax.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()


def get_3D_volume(path, opacity=1, color="red"):
    enable_plotly_in_cell()
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
  

def plot_streamlines(path, ref_img_path, subsampling=10):
    enable_plotly_in_cell()

    affine = nib.load(ref_img_path).affine

    streams, hdr = trackvis.read(path)
    streamlines = [s[0] for s in streams]
    streamlines = list(move_streamlines(streamlines, np.linalg.inv(affine)))

    traces = []
    for sl in streamlines[::subsampling]:
        color = get_voxelwise_orientation_colormap([sl], orientation="saggital")[0]
        x,y,z, = zip(*sl)
        trace = go.Scatter3d(
            x=x, y=y, z=z,
            line=dict(
                color=color,
                width=2
            ),
            mode="lines"
        )
        traces.append(trace)
    return traces


def plot_streamlines_with_brain_mask_and_endings_mask(sl_path, brain_mask_path, endings_path,
                                                      subsampling=10):
    brain_mask = get_3D_volume(brain_mask_path, opacity=0.1, color="silver")
    endings_mask = get_3D_volume(endings_path, opacity=0.2, color="salmon")
    t = get_streamlines_plot(sl_path,brain_mask_path, subsampling=subsampling)
    
    layout=go.Layout(showlegend=False)
    figure=go.Figure(data=[brain_mask, endings_mask] + t, layout=layout)
    plotly.offline.iplot(figure)

    
def merge_masks(path_in_1, path_in_2, path_out):

    ref_img = nib.load(path_in_1)  # use first mask as ref image

    new_mask = np.zeros(ref_img.get_data().shape)
    for file in [path_in_1, path_in_2]:
        mask = nib.load(file)
        mask_data = mask.get_data()
        new_mask[mask_data > 0] = 1

    combined_img = nib.Nifti1Image(new_mask.astype("uint8"), ref_img.affine) 
    nib.save(combined_img, path_out)


