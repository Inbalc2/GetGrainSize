import os
import cv2
from collections import defaultdict
from PIL import Image
import numpy as np
from cv_algorithms import guo_hall

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def map_images_by_index(images_path):
    # Create a dictionary mapping image indices to their corresponding filenames
    images = [f'{images_path}/{f}' for f in os.listdir(images_path) if f.endswith('png')]
    mapper = defaultdict(list)
    for im in images:
        im_index = int(im.split('/')[-1].split('-')[0])
        mapper[im_index].append(im)
    return mapper

def crop_and_save_images(full_image_path, zones, output_path):
    full_image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    if full_image is None:
        print(f"Error loading image: {full_image_path}")
        return

    for zone in zones:
        # Parse the coordinates from the zone filename
        coords = os.path.basename(zone).split('.')[0].split('-')[1:]
        if len(coords) != 4:
            print(f"Invalid zone format for file {zone}. Skipping...")
            continue

        y, x, sy, sx = [int(e) for e in coords]
        # Crop the specified zone from the full image
        crop = full_image[y + sy:y + sy + 128, x + sx:x + sx + 128]
        output_filename = f'{output_path}/{os.path.basename(zone)}'
        cv2.imwrite(output_filename, crop)

def unify_crops(input_dir, output_dir, crop_size=256, unified_size=128):
    create_directory(output_dir)  
    files = sorted(os.listdir(input_dir))
    for i in range(0, len(files), 4):
        # Create a new image to hold the unified crops
        unified_image = Image.new('RGB', (crop_size, crop_size), (0, 0, 0))
        for j in range(4):
            crop_image = Image.open(os.path.join(input_dir, files[i + j]))
            x = (j % 2) * unified_size
            y = (j // 2) * unified_size
            unified_image.paste(crop_image, (x, y))

        new_filename = '-'.join(files[i].split('-')[:3]) + '.png'
        unified_image.save(os.path.join(output_dir, new_filename))

def process_images(clemex_path, zones_path, output_path, full_image_ext='png'):
    #Process and crop images based on the zones.
    create_directory(output_path)
    mapper = map_images_by_index(zones_path)
    for k in mapper:
        full_image_path = f'{clemex_path}/{k}.{full_image_ext}'
        if not os.path.exists(full_image_path):
            print(f"File {full_image_path} not found. Skipping...")
            continue
        # Crop and save images from the full image based on the zones
        crop_and_save_images(full_image_path, mapper[k], output_path)


def apply_guo_hall_thinning(image):
    image = np.array(image)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    binary_image = cv2.bitwise_not(binary_image)
    thinned_image = guo_hall(binary_image, inplace=False)
    thinned_image = cv2.bitwise_not(thinned_image)
    return thinned_image

def convert_to_color(thinned_image, original_image):
    color_thinned_image = cv2.cvtColor(thinned_image, cv2.COLOR_GRAY2RGB)
    combined_image = np.where(thinned_image[..., None] == 0, original_image, color_thinned_image)
    return combined_image

def process_img(input_dir, output_dir):
    # Process each image in the input directory and save the thinned image
    for image_name in os.listdir(input_dir):
        if image_name.endswith(('.png', '.jpg', '.tif')):
            input_path = os.path.join(input_dir, image_name)
            print(f"Processing image: {input_path}")
            original_image = Image.open(input_path)
            grayscale_image = original_image.convert('L')  # Convert to grayscale
            # Apply Guo-Hall thinning
            thinned_image = apply_guo_hall_thinning(grayscale_image)
            original_image_np = np.array(original_image)
            thinned_image_color = convert_to_color(thinned_image, original_image_np)
            final_image = Image.fromarray(thinned_image_color.astype(np.uint8))
            output_path = os.path.join(output_dir, image_name)
            final_image.save(output_path)
            print(f"Saved processed image: {output_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and unify images for metallography.")
    parser.add_argument('--clemex_path', type=str, required=True, help='Path to the clemex predictions directory.')
    parser.add_argument('--zones_path', type=str, required=True, help='Path to crops with the zones in their filename.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory.')
    parser.add_argument('--mlography_path', type=str, required=True, help='Path to the metalography predictions directory.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the ground truth images directory of crops 128x128.')
    
    args = parser.parse_args()

    clemex_path = args.clemex_path
    zones_path = args.zones_path
    output_path = args.output_path
    mlography_path = args.mlography_path
    gt_path = args.gt_path

    # Process Clemex predictions and create unified crops
    clemex_128_out = os.path.join(output_path, 'clemex_predictions_squares_128')
    clemex_256_path = os.path.join(output_path, 'clemex_unified_crops_256')
    process_images(clemex_path, zones_path, clemex_128_out)
    create_directory(clemex_256_path)
    unify_crops(clemex_128_out, clemex_256_path)

    # Process Mlography predictions and create unified crops
    mlography_128_out = os.path.join(output_path, 'mlography_predictions_squares_128')
    mlography_256_path = os.path.join(output_path, "mlography_predictions_unified_crops_256")
    process_images(mlography_path, zones_path, mlography_128_out)
    create_directory(mlography_256_path)
    unify_crops(mlography_128_out, mlography_256_path)

    # Process Ground Truth (GT) images and create unified crops
    gt_256_path = os.path.join(output_path, "GT_unified_crops_256")
    create_directory(gt_256_path)
    unify_crops(gt_path, gt_256_path)
    
    # Apply Guo-Hall thinning to the masks 
    process_img(clemex_256_path, clemex_256_path)
    process_img(gt_256_path, gt_256_path)

if __name__ == "__main__":
    main()
