'''
Very simple Python script for visualizing kw18 images
'''
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import click
import os
from ubelt.util_path import ensuredir
import numpy as np

# https://gitlab.kitware.com/vigilant/ovharn.git
from ovharn.util.kw18 import read_kw18

def read_in_image(fpath):
    # TODO: use netharn or kwimage or other more flexible
    # reading option
    return rescale_intensity_01(
            cv2.imread(fpath, cv2.IMREAD_UNCHANGED),
            (0,255))

def rescale_intensity_01(image, in_range, scale='linear'):
    """
    This code has been copied from Jon Crall's 
    https://gitlab.kitware.com/OPIR/irharn/-/blob/master/irharn/ir_frames.py#L77

    simplified version of skimage implementation with optional log scaling.


    Args:
        image (np.ndarray): image to normalize
        in_range (tuple): minimum and maximum values in the image

    Example:
        >>> rng = np.random.RandomState(0)
        >>> image = rng.rand(4, 4) * 100
        >>> in_range = (0, 100)
        >>> image = rescale_intensity_01(image, in_range)
        >>> assert image.min() >= 0 and image.max() <= 1

    Example:
        >>> rng = np.random.RandomState(0)
        >>> image = rng.rand(4, 4) * 100
        >>> in_range = 'data'
        >>> image = rescale_intensity_01(image, in_range)
        >>> assert image.min() == 0 and image.max() == 1

    Doctest:
        >>> rng = np.random.RandomState(0)
        >>> image = rng.rand(4, 4) * 100
        >>> lin01 = rescale_intensity_01(image, (0, 100), 'linear')
        >>> log01 = rescale_intensity_01(image, (0, 100), 'log')
        >>> x = rescale_intensity_01(lin01, (0, 1), 'log')
    """
    return image
    if in_range == 'data':
        imin = image.min()
        imax = image.max()
    else:
        imin, imax = in_range

    image = image.astype(np.float32).clip(imin, imax)

    if scale == 'linear':
        image = (image - imin) / (imax - imin)
    elif scale == 'log':
        image = np.log(image - (imin - 1)) / np.log(imax - (imin - 1))
    else:
        raise KeyError(scale)
    return image



@click.command()
@click.option(
    '--kw18_fpath',
    default=None,
    help='Filepath to a kw18 file',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False
    )
)
@click.option(
    '--img_dir',
    default=None,
    help='Directory to images',
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True
    )
)
@click.option(
    '--num_tracks_to_show',
    default=None,
    help='Number of tracks to show',
    type=int
)
@click.option(
    '--num_images_to_show',
    default=None,
    help='Number of images to show',
    type=int
)
@click.option(
    '--show',
    is_flag=True,
    help='If present, will show, otherwise will write to disk'
)
def main(kw18_fpath, img_dir, num_tracks_to_show = None, num_images_to_show = None, show = True):
    kw18_ = read_kw18(kw18_fpath).sort_values('track_id')
    
    # assume that the images are sorted by name
    # TODO: add extra tooling around this given
    # frame number from kw18 file for more flexibility
    images_fpaths = sorted(glob.glob(os.path.join(img_dir,'*')))

    frame_subtractor = min(kw18_['frame_number'])

    all_track_ids = list(set(kw18_['track_id']))
    if num_tracks_to_show is None or num_tracks_to_show > len(all_track_ids):
        num_tracks_to_show = len(all_track_ids)

    all_images = {}
    
    for i in range(num_tracks_to_show):
        right_track = kw18_.loc[kw18_['track_id'] == all_track_ids[i]]
        num_images_copy = num_images_to_show
        if num_images_copy is None or num_images_copy > len(right_track):
            num_images_copy = len(right_track)

        for n in range(num_images_copy):
            row = right_track.iloc[n]
            #while
            frame_number = int(row['frame_number'] - frame_subtractor)
            
            if frame_number > len(images_fpaths):
                # would only reach here if there are more frames in kw18
                # in there are images in the image directory
                # Probably not the best way to handle this as is -
                # at least throw a warning
                continue

            # TODO: more elegantly manage reading in lots of large images
            image_fpath = images_fpaths[frame_number]
            

            if frame_number in all_images:
                img = all_images[frame_number]
                print("adding")
            else:
                print('Reading in {}'.format(image_fpath))
                img = read_in_image(image_fpath)
                
            x,y = int(row['image_loc_x']), int(row['image_loc_y'])
            area = 5 # amnually encoded although can pull this from kw18
            img = cv2.circle(img, (x,y), area, color = (0,0,255), thickness = 2)
            all_images[frame_number] = img

    if not show:
        output_dir = 'track_outputs'
        ensuredir(output_dir)
    
    sorted_keys = sorted(all_images.keys())
    for frame_number in sorted_keys:
        img = all_images[frame_number]
        if show:
            # TODO: has some serious memory concerns
            plt.imshow(img)
            plt.pause(1)
        else:
            
            cv2.imwrite("{}/img_{}.png".format(output_dir,frame_number),img) 

if __name__ == "__main__":
    main()
