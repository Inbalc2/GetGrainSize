import cv2
import numpy as np
import os



# # Define the input directory and output directory
# input_dir = '/Users/inbal/Desktop/Metallography_2/clemex_predictions'
# output_dir = '/Users/inbal/Desktop/Metallography_2/clemex_predictions_modified'

# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # List all files in the input directory
# files = os.listdir(input_dir)

# # Iterate over each file in the input directory
# for file in files:
#     # Read the image
#     img = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_GRAYSCALE)

#     # Check if the image is not empty
#     if img is not None:
#         # Threshold the image to get a binary image
#         _, binary_img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

#         # Invert the binary image (foreground becomes black, background becomes white)
#         inverted_img = cv2.bitwise_not(binary_img)

#         # Save the inverted image to the output directory
#         cv2.imwrite(os.path.join(output_dir, file), inverted_img)
#     else:
#         print(f"Failed to read image at {os.path.join(input_dir, file)}")





# def remove_red_lines_and_blue_dots(input_folder, output_folder):
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # List all files in the input folder
#     files = os.listdir(input_folder)

#     for file in files:
#         if file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
#             input_image_path = os.path.join(input_folder, file)

#             # Load the image
#             img = cv2.imread(input_image_path)

#             # Check if image was successfully loaded
#             if img is None:
#                 print(f"Error: Unable to load image '{input_image_path}'")
#                 continue

#             # Convert image to HSV color space
#             hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#             # Define lower and upper bounds for the red color in HSV space
#             lower_red1 = np.array([0, 50, 50])
#             upper_red1 = np.array([10, 255, 255])
#             lower_red2 = np.array([170, 50, 50])
#             upper_red2 = np.array([180, 255, 255])

#             # Create masks for red color
#             mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#             mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#             red_mask = cv2.bitwise_or(mask1, mask2)

#             # Define lower and upper bounds for the blue color in HSV space
#             lower_blue = np.array([100, 50, 50])
#             upper_blue = np.array([140, 255, 255])

#             # Create a mask for blue color
#             blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

#             # Combine the red and blue masks
#             combined_mask = cv2.bitwise_or(red_mask, blue_mask)

#             # Inpaint the red lines and blue dots
#             inpainted_img = cv2.inpaint(img, combined_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

#             # Remove vertical and horizontal lines
#             kernel = np.ones((3, 3), np.uint8)
#             inpainted_img = cv2.morphologyEx(inpainted_img, cv2.MORPH_OPEN, kernel)

#             # Save the inpainted image
#             output_path = os.path.join(output_folder, file)
#             cv2.imwrite(output_path, inpainted_img)

#             # Debugging print statements
#             print(f"Processed: {input_image_path} -> Saved as: {output_path}")

# # Activate the function on the specified directory
# input_folder = "/Users/inbal/Desktop/Metallography_2/metalography_predictions"
# output_folder = "/Users/inbal/Desktop/Metallography_2/metalography_predictions" # Save the processed images in the same folder

# #remove_red_lines_and_blue_dots(input_folder, output_folder)

import os
import cv2
from collections import defaultdict
from pprint import pprint
import os
from PIL import Image
import numpy as np
import cv2

p_clemex = '/Users/inbal/Desktop/Metallography_2/clemex_predictions_modified'
p_images_by_zone = '/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/image'
output_path = '/Users/inbal/Desktop/Metallography_2/clemex_predictions_squares_128'

if not os.path.exists(output_path):
   os.makedirs(output_path, exist_ok=True)

clemex_images = [f for f in os.listdir(p_clemex) if f.endswith('png')]
zone_images =[f'{p_images_by_zone}/{f}' for f in os.listdir(p_images_by_zone) if f.endswith('png')]

mapper = defaultdict(list)
for im in zone_images:
   im_index = int(im.split('/')[-1].split('-')[0])
   mapper[im_index].append(im)

# pprint(mapper)

for k in mapper:
   _clemex_im_path = f'{p_clemex}/{k}.png'
   full_clemex = cv2.imread(_clemex_im_path, cv2.IMREAD_GRAYSCALE)
   # print(full_clemex.shape)
   k_zones = mapper[k]

   for z in k_zones:
      _curr_zone = os.path.basename(z).split('.')[0].split('-')[1:]
      y, x, sy, sx = [int(e) for e in _curr_zone]
      _box = full_clemex[y+sy:y+sy+128, x+sx:x+sx+128]
      p_name = os.path.basename(z)
      cv2.imwrite(f'{output_path}/{p_name}', _box)



clemex_256_crops_predictions_path = "/Users/inbal/Desktop/Metallography_2/clemex_unified_crops_256_fixed"
if not os.path.exists(clemex_256_crops_predictions_path):
   os.makedirs(clemex_256_crops_predictions_path, exist_ok=True)

def unify_crops(input_dir, output_dir):
    # List all files in the input directory
    files = os.listdir(input_dir)
    files = sorted(files)

    # Iterate over files in groups of 4
    for i in range(0, len(files), 4):
        # Create a new blank image
        unified_image = Image.new('RGB', (256, 256), (0, 0, 0))  # Start with white background

        # Iterate over the 4 files in the group
        for j in range(4):
            # Open each crop image
            crop_image = Image.open(os.path.join(input_dir, files[i + j]))

            # Calculate the position to paste the crop
            x = (j % 2) * 128
            y = (j // 2) * 128

            # Paste the crop onto the unified image
            unified_image.paste(crop_image, (x, y))

        # Save the unified image with the appropriate filename
        new_filename = files[i].split('-')[:3]  # Take first 3 parts of filename
        new_filename = '-'.join(new_filename) + '.png'
        unified_image.save(os.path.join(output_dir, new_filename))

#unify_crops("/Users/inbal/Desktop/Metallography_2/mlography_predictions_squares_128", "/Users/inbal/Desktop/Metallography_2/mlography_predictions_squares_256_unified")

p_images_by_zone = '/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/image'
p_mlography = '/Users/inbal/Desktop/Metallography_2/metalography_predictions'
output_path_mlography = '/Users/inbal/Desktop/Metallography_2/mlography_predictions_squares_128'

if not os.path.exists(output_path_mlography):
   os.makedirs(output_path_mlography, exist_ok=True)

mlography_images = [f for f in os.listdir(p_mlography) if f.endswith('png')]
zone_images =[f'{p_images_by_zone}/{f}' for f in os.listdir(p_images_by_zone) if f.endswith('png')]

mapper_mlography = defaultdict(list)
for im in zone_images:
   im_index = int(im.split('/')[-1].split('-')[0])
   mapper_mlography[im_index].append(im)

# Crop and save images
for k in mapper_mlography:
    _mlography_im_path = f'{p_mlography}/{k}.png'  # Change the file extension to '.tif'
    if not os.path.exists(_mlography_im_path):
        print(f"File {_mlography_im_path} not found. Skipping...")
        continue
    
    full_mlography = cv2.imread(_mlography_im_path, cv2.IMREAD_GRAYSCALE)

    k_zones = mapper_mlography[k]

    for z in k_zones:
        _curr_zone = os.path.basename(z).split('.')[0].split('-')[1:]
        if len(_curr_zone) != 4:
            print(f"Invalid zone format for file {z}. Skipping...")
            continue

        y, x, sy, sx = [int(e) for e in _curr_zone]
        _box = full_mlography[y+sy:y+sy+128, x+sx:x+sx+128]
        p_name = os.path.basename(z)
        cv2.imwrite(f'{output_path_mlography}/{p_name}', _box)

##############################
mlography_256_crops_predictions_path = "/Users/inbal/Desktop/Metallography_2/mlography_prediction_unified_crops_256"
if not os.path.exists(mlography_256_crops_predictions_path):
   os.makedirs(mlography_256_crops_predictions_path, exist_ok=True)
unify_crops(output_path_mlography, mlography_256_crops_predictions_path)

unify_crops("/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/label", "/Users/inbal/Desktop/Metallography_2/GT_256_crops")






###################
import cv2
import os
from collections import defaultdict

def extract_info_from_filename(filename):
    try:
        parts = filename.split('-')
        modelname = parts[0]
        x = int(parts[1])
        y = int(parts[2])
        dx = int(parts[3])
        dy = int(parts[4].split('.')[0])  # Assuming the format includes .png after dy
        return modelname, x, y, dx, dy
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None, None, None, None, None

def parse_gt_files(gt_image_dir):
    gt_zones = defaultdict(set)
    gt_files = [f for f in os.listdir(gt_image_dir) if f.endswith('.png')]
    for f in gt_files:
        modelname, x, y, dx, dy = extract_info_from_filename(f)
        if None not in (modelname, x, y, dx, dy):
            gt_zones[modelname].add((x, y))
    return gt_zones

def x_y_in_gt(x, y, delta, gt_zones, modelname):
    for (x_gt, y_gt) in gt_zones[modelname]:
        if (x <= x_gt <= x + delta) and (y <= y_gt <= y + delta):
            return True
    return False

def crop_images(image_dir, output_dir, zone_size=(256, 256), gt_image_dir='/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/inv_label'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse ground truth files
    gt_zones = parse_gt_files(gt_image_dir)

    # List all files in the directory
    files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    print(f"Total files found: {len(files)}")

    crop_counts = defaultdict(int)

    for filename in files:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_height, img_width = img.shape[:2]
        modelname = filename.split('.')[0]

        for y in range(0, img_height, zone_size[1]):
            for x in range(0, img_width, zone_size[0]):
                if crop_counts[modelname] >= 4:
                    break  # Stop after getting 4 crops per model

                if x_y_in_gt(x, y, zone_size[0], gt_zones, modelname):
                    print(f"Skipping zone {modelname}-{x}-{y} as it overlaps with GT zones.")
                    continue  # Skip this zone if it overlaps with any GT zones

                crop_img = img[y:y + zone_size[1], x:x + zone_size[0]]
                if crop_img.shape[0] != zone_size[1] or crop_img.shape[1] != zone_size[0]:
                    continue  # Skip incomplete zones

                output_filename = f"{modelname}-{x}-{y}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, crop_img)
                crop_counts[modelname] += 1
                print(f"Saved {output_path}")

# Define directories and their configurations
image_dir = '/Users/inbal/Desktop/Metallography_2/metalography_predictions'
output_dir = '/Users/inbal/Desktop/Metallography_2/mlography_crops_256_hyen_check_last'
gt_image_dir = '/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/inv_label'

# Crop images avoiding specified zones and ensuring 4 crops per model
crop_images(image_dir, output_dir, gt_image_dir=gt_image_dir)

################################################

from PIL import Image
import numpy as np
import cv2
from cv_algorithms import guo_hall


# Paths
clemex_input_dir = '/Users/inbal/Desktop/Metallography_2/clemex_unified_crops_256_fixed'
clemex_output_dir = '/Users/inbal/Desktop/Metallography_2/clemex_unified_crops_256_fixed'
# Ensure the output directories exist


def apply_guo_hall_thinning(image):
    # Convert PIL image to numpy array
    image = np.array(image)
    
    # Convert to binary image (0 or 255)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image
    binary_image = cv2.bitwise_not(binary_image)
    
    # Apply Guo-Hall thinning
    thinned_image = guo_hall(binary_image, inplace=False)
    
    # Invert the thinned image back to original colors
    thinned_image = cv2.bitwise_not(thinned_image)
    
    return thinned_image

def convert_to_color(thinned_image, original_image):
    # Convert thinned image to color (RGB) format using the original image
    color_thinned_image = cv2.cvtColor(thinned_image, cv2.COLOR_GRAY2RGB)
    
    # Combine the color thinned image with the original image
    combined_image = np.where(thinned_image[..., None] == 0, original_image, color_thinned_image)
    
    return combined_image

# Process each image in the input directory and save the thinned images
def process_images(input_dir, output_dir):
    for image_name in os.listdir(input_dir):
        if image_name.endswith(('.png', '.jpg', '.tif')):
            input_path = os.path.join(input_dir, image_name)
            print(f"Processing image: {input_path}")
            
            # Open image
            original_image = Image.open(input_path)
            grayscale_image = original_image.convert('L')  # Convert to grayscale

            # Apply Guo-Hall thinning
            thinned_image = apply_guo_hall_thinning(grayscale_image)
            
            # Convert to color
            original_image_np = np.array(original_image)
            thinned_image_color = convert_to_color(thinned_image, original_image_np)

            # Convert back to PIL image
            final_image = Image.fromarray(thinned_image_color.astype(np.uint8))

            # Save the processed image
            output_path = os.path.join(output_dir, image_name)
            final_image.save(output_path)
            print(f"Saved processed image: {output_path}")

print("Processing Clemex images...")
process_images(clemex_input_dir, clemex_output_dir)

