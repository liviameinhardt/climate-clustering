#%%

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
import cv2 



def open_nc(file_path):

    try:
        ds = xr.open_dataset(file_path)

    except Exception as e:
        print(f"Error opening file: {e}")
        return 

    else:
        return ds
    
def get_data(file_path,variable,quantile=False,wind=1):

    df = open_nc(file_path)
    df = df[variable].values


    if quantile:
        max_o = np.quantile(df,0.99)
        min_o = np.quantile(df,0.1)

        df = np.clip(df, min_o, max_o)

    else:
        min_o = np.min(df)
        max_o = np.max(df)

    norm_imgs = (df - min_o)/(max_o -min_o)

    time, lat, long = norm_imgs.shape

    norm_imgs = norm_imgs.reshape(int(time/wind), wind, lat, long)
    norm_imgs = np.transpose(norm_imgs, (0, 2, 3, 1))

    if wind==1:
        norm_imgs = norm_imgs.squeeze() 

    return norm_imgs

def extract_patches(image_tensor, patch_size=(32, 32), num_patches=15, overlap=4):

    height, width = image_tensor.shape
    patch_height, patch_width = patch_size

    # Calculate valid range for patch centers
    valid_y = height - patch_height
    valid_x = width - patch_width

    # Store the centers of the patches to ensure low overlap
    centers = []

    for _ in range(num_patches):

        while True:
            center_y = random.randint(0, valid_y)
            center_x = random.randint(0, valid_x)
            new_center = (center_y, center_x)

            # # Check for overlap with existing centers
            if all(abs(center_y - c[0]) >= overlap and abs(center_x - c[1]) >= overlap for c in centers)\
                and ((center_y<=100 ) and (center_x<=90)):
                
                centers.append(new_center)
                break

    patches = []
    for center in centers:
        y, x = center
        patch = image_tensor[ y:y + patch_height, x:x + patch_width]
        patches.append(patch)

    return patches, centers

def get_data_in_patches(data,patch_size=(32, 32), num_patches=15, overlap=4):

    final_data = []
    for idx in range(len(data)):

        patches, _ = extract_patches(data[idx],patch_size,num_patches,overlap)
        final_data+=patches

    return np.array(final_data)


def highlight_patches(original_image, centers, patch_size=(32, 32)):
    
    
    patch_height, patch_width = patch_size

    # Convert the image to 0-255 range for OpenCV
    original_image_bgr = (original_image * 255).astype(np.uint8)

    for center in centers:
        y, x = center
        # Draw a rectangle around the patch using OpenCV
        cv2.rectangle(original_image_bgr, (x, y), (x + patch_width, y + patch_height), (255, 0, 0), 1)

    plt.imshow(original_image_bgr / 255.0)  # Convert back to [0, 1] for displaying
    plt.axis('off')
    plt.title('Highlighted Patches')
    plt.show()


# #get patches 32x32
data = get_data("data/era_t2m_msl_1983_2023.nc",variable='t2m')
data = get_data_in_patches(data,patch_size=(40, 40), num_patches=10, overlap=4)
np.save("data/training_patches/temperature.npy",data,allow_pickle=False)