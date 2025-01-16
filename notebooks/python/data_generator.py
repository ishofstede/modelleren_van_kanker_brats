# https://youtu.be/PNqnLbzdxwQ
"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template to create your own custom data generators. 

No image processing operations are performed here, just load data from local directory
in batches. 

"""
import numpy as np
import os 

class DataGenerator:
    @staticmethod
    def load_img(img_dir, img_list):
        images = []
        for i, image_name in enumerate(img_list):    
            if image_name.split('.')[1] == 'npy':
                image = np.load(os.path.join(img_dir, image_name))
                images.append(image)
        images = np.array(images, dtype=np.float32)  # Ensure data type is float32
        return images

    @staticmethod
    def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
        L = len(img_list)

        # keras needs the generator infinite, so we will use while true  
        while True:
            batch_start = 0
            batch_end = batch_size

            while batch_start < L:
                limit = min(batch_end, L)

                # Call the class method explicitly
                X = DataGenerator.load_img(img_dir, img_list[batch_start:limit])
                Y = DataGenerator.load_img(mask_dir, mask_list[batch_start:limit])

                # Cast to float32 to ensure consistency
                X = X.astype(np.float32)
                Y = Y.astype(np.float32)

                yield (X, Y)  # a tuple with two numpy arrays with batch_size samples

                batch_start += batch_size   
                batch_end += batch_size

