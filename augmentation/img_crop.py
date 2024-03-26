import os
from PIL import Image
import argparse


def read_bounding_boxes(file_path):
    
    """Read bounding boxes from bounding_boxes.txt

    Returns:
        dictionary: bounding boxes coordinates
    """
    
    bounding_boxes = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            bounding_boxes[parts[0]] = list(map(float, parts[1:]))
    return bounding_boxes


def read_image_id(file_path):
    image_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            image_id = parts[0]
            image_path = parts[1]

            # Extract the image name from the path
            _, image_name = os.path.split(image_path)

            # Remove leading numbers from the image name (if present)
            image_name = ''.join([i for i in image_name if not i.isdigit()])

            # Add the entry to the dictionary
            image_dict[image_name] = image_id

    # Print the resulting dictionary
    print(image_dict)


def crop_images(images_folder, output_folder, bounding_boxes):
    
    """crop images based on bounding boxes
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        

    for file_name in os.listdir(images_folder):
        if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Add more image extensions if needed
            
            print("processing")
            
            image_id = file_name.split('.')[0]
            
            print(image_id)
            
            if image_id in bounding_boxes:
                box = bounding_boxes[image_id]
                image_path = os.path.join(images_folder, file_name)
                output_path = os.path.join(output_folder, file_name)

                with Image.open(image_path) as img:
                    cropped_img = img.crop(box)
                    cropped_img.save(output_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bounding_boxes_file', type=str, default='./datasets/bounding_boxes.txt', 
                        help='txt file for bounding box')
    
    parser.add_argument('--imgid_file', type=str, default='./datasets/images.txt', 
                        help='txt file for image id')
    
    parser.add_argument('--images_folder', type=str, default='./datasets/cub200_cropped/', 
                        help='Original Uncropped Images Folder')
    
    parser.add_argument('--output_folder', type=str, default='./datasets/cub200_cropped/')
    
    args = parser.parse_args()
    
    bounding_boxes = read_bounding_boxes(args.bounding_boxes_file)

    img_id = read_image_id(args.imgid_file)
    
    crop_images(args.images_folder, args.output_folder, args.bounding_boxes)