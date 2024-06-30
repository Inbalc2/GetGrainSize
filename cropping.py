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
        coords = os.path.basename(zone).split('.')[0].split('-')[1:]
        if len(coords) != 4:
            print(f"Invalid zone format for file {zone}. Skipping...")
            continue

        y, x, sy, sx = [int(e) for e in coords]
        crop = full_image[y + sy:y + sy + 128, x + sx:x + sx + 128]
        output_filename = f'{output_path}/{os.path.basename(zone)}'
        cv2.imwrite(output_filename, crop)

def unify_crops(input_dir, output_dir, crop_size=256, unified_size=128):
    create_directory(output_dir)  # Ensure the output directory is created
    files = sorted(os.listdir(input_dir))
    for i in range(0, len(files), 4):
        unified_image = Image.new('RGB', (crop_size, crop_size), (0, 0, 0))
        for j in range(4):
            crop_image = Image.open(os.path.join(input_dir, files[i + j]))
            x = (j % 2) * unified_size
            y = (j // 2) * unified_size
            unified_image.paste(crop_image, (x, y))

        new_filename = '-'.join(files[i].split('-')[:3]) + '.png'
        unified_image.save(os.path.join(output_dir, new_filename))

def process_images(clemex_path, zones_path, output_path, full_image_ext='png'):
    create_directory(output_path)
    mapper = map_images_by_index(zones_path)
    for k in mapper:
        full_image_path = f'{clemex_path}/{k}.{full_image_ext}'
        if not os.path.exists(full_image_path):
            print(f"File {full_image_path} not found. Skipping...")
            continue
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

# Process each image in the input directory and save the thinned images
def process_img(input_dir, output_dir):
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
    clemex_path = '/Users/inbal/Desktop/Metallography_2/clemex_predictions_modified'
    zones_path = '/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/image'
    output_path = '/Users/inbal/Desktop/Metallography_2/clemex_predictions_squares_128_'
    process_images(clemex_path, zones_path, output_path)

    clemex_256_path = '/Users/inbal/Desktop/Metallography_2/clemex_unified_crops_256_'
    create_directory(clemex_256_path)
    unify_crops(output_path, clemex_256_path)

    mlography_path = '/Users/inbal/Desktop/Metallography_2/metalography_predictions'
    output_path_mlography = '/Users/inbal/Desktop/Metallography_2/mlography_predictions_squares_128_'
    process_images(mlography_path, zones_path, output_path_mlography)

    mlography_256_path = "/Users/inbal/Desktop/Metallography_2/mlography_predictions_unified_crops_256_"
    create_directory(mlography_256_path)
    unify_crops(output_path_mlography, mlography_256_path)

    gt_path = "/Users/inbal/Desktop/Metallography_2/MLography/Segmentation/unet/data/squares_128/train/label"
    gt_output_path = "/Users/inbal/Desktop/Metallography_2/GT_256_crops_"
    create_directory(gt_output_path)  # Ensure the output directory is created
    unify_crops(gt_path, gt_output_path)

    process_img(clemex_256_path,clemex_256_path)
    process_img(gt_output_path,gt_output_path)

if __name__ == "__main__":
    main()
