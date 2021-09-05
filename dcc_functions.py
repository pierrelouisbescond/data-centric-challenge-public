import numpy as np
import glob
import cv2
from PIL import Image, ImageOps


def how_many_files_in_folder(path):
    files = glob.glob(path)
    nb_files = len(files)
    return nb_files, files


class FOLDER():
    
    def __init__(self, DATA_FOLDER, FOLDERS, LABELS, ):
        self.DATA_FOLDER = DATA_FOLDER
        self.FOLDERS = FOLDERS
        self.LABELS = LABELS

    def summary(self, display_ratio=4):
        
        for folder in self.FOLDERS:

            sum_folder = 0

            for label in self.LABELS:

                path = self.DATA_FOLDER+folder+"/"+label+"/*.png"
                
                nb_files = len(glob.glob(path))

                sum_folder += nb_files

                print(folder.ljust(10), ":", label.ljust(4), ":", nb_files, ":", "*" * (nb_files // display_ratio))

            print("Total Number of pictures in", folder, ":", sum_folder, "\n")

            if folder == "train":
                folder_size_train = sum_folder
            elif folder == "val":
                folder_size_val = sum_folder

        print(f"train/val ratio: {100 * (folder_size_train) / (folder_size_train+folder_size_val):.1f} %")


def translate_picture(source_image, x_range, y_range):
    translation_matrix = np.float32([[1, 0, np.random.randint(-x_range, x_range)],
                                     [0, 1, np.random.randint(-y_range, y_range)]])

    translated_img_data = cv2.warpAffine(source_image, translation_matrix, (source_image.shape[1], source_image.shape[0]), cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    translated_img = Image.fromarray(translated_img_data)
    
    return translated_img


            
def crop_and_square_image(image, padding=0.01):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nb_rows, nb_cols = gray_image.shape
    
    non_empty_columns = np.where(gray_image.min(axis=0) < 255)[0]
    non_empty_rows = np.where(gray_image.min(axis=1) < 255)[0]
    
    cropBox = (int(min(non_empty_rows) * (1 - padding)),
               int(min(max(non_empty_rows) * (1 + padding), nb_rows)),
               int(min(non_empty_columns) * (1 - padding)),
               int(min(max(non_empty_columns) * (1 + padding), nb_cols)))
    
    cropped_image = image[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1, :]

    # After cropping the image, we reconstruct it as a square to make sure
    # that the 32 x 32 transformation will not distort it
    
    max_dimension = max(cropped_image.shape)

    extra_heigth = (max_dimension-cropped_image.shape[0])//2
    extra_width = (max_dimension-cropped_image.shape[1])//2
    
    cropped_and_squared_image_data = cv2.copyMakeBorder(cropped_image, extra_heigth, extra_heigth, extra_width, extra_width, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    
    cropped_and_squared_image = Image.fromarray(cropped_and_squared_image_data)

    return cropped_and_squared_image


def add_sp_noise(img, wpixels_max=5000, bpixels_max=5000):
  
    # Getting the dimensions of the image
    row, col = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    number_of_pixels = np.random.randint(100, wpixels_max)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord = np.random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord = np.random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = np.random.randint(300, bpixels_max)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord = np.random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord = np.random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_coord][x_coord] = 0
    
    img_noise = Image.fromarray(img)

    return img_noise

def add_background_noise(source_path, noise_path):

    source_image = Image.open(source_path)
    
    noise_image = Image.open(noise_path).resize(source_image.size)

    random_choice = np.random.randint(1, 5)
    
    if random_choice == 2:
        
        noise_image = noise_image.transpose(Image.FLIP_TOP_BOTTOM)
    
    elif random_choice == 3:
        
        noise_image = ImageOps.mirror(noise_image)
    
    elif random_choice == 4:
        
        noise_image = noise_image.transpose(Image.FLIP_TOP_BOTTOM)
        noise_image = ImageOps.mirror(noise_image)
    
    mask = Image.new("L", source_image.size, 128)
    noisy_image = Image.composite(source_image, noise_image, mask)
    
    noisy_image_array = np.array(noisy_image)
    
    noisy_image_array = ((noisy_image_array - noisy_image_array.min()) * (1/(noisy_image_array.max() - noisy_image_array.min()) * 254))
    
    noisy_image = Image.fromarray(noisy_image_array).convert("L")
    
    return noisy_image, random_choice