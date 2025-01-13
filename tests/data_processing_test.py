import unittest
import numpy as np
import nibabel as nib
import os
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from src.data_processing import (  
    load_and_normalize_image,
    load_and_process_mask,
    crop_volume,
    save_volume_and_mask,
    process_and_save_images
)

scaler = MinMaxScaler()

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        """
        Create mock data for testing.
        """
        # Mock 3D image and mask
        self.image = np.random.rand(240, 240, 155) * 1000  # Random 3D image
        self.mask = np.random.randint(0, 5, (240, 240, 155))  # Random labels (0-4)
        self.save_path = 'test_output'
        self.img_idx = 0

        # Expected crop parameters
        self.crop_size = (128, 128, 128)
        self.offsets = (56, 56, 13)

    def tearDown(self):
        """
        Clean up test outputs.
        """
        if os.path.exists(self.save_path):
            for root, _, files in os.walk(self.save_path):
                for file in files:
                    os.remove(os.path.join(root, file))
                os.rmdir(root)

    def test_load_and_normalize_image(self):
        """
        Test normalization of a 3D image.
        """
        nib.save(nib.Nifti1Image(self.image, np.eye(4)), 'test_image.nii')
        normalized_image = load_and_normalize_image('test_image.nii')
        self.assertTrue(np.allclose(normalized_image.min(), 0))
        self.assertTrue(np.allclose(normalized_image.max(), 1))
        os.remove('test_image.nii')

    def test_load_and_process_mask(self):
        """
        Test loading and label reassignment in a mask.
        """
        nib.save(nib.Nifti1Image(self.mask, np.eye(4)), 'test_mask.nii')
        processed_mask = load_and_process_mask('test_mask.nii')
        self.assertNotIn(4, np.unique(processed_mask))
        self.assertIn(3, np.unique(processed_mask))
        os.remove('test_mask.nii')

    def test_crop_volume(self):
        """
        Test cropping of a 3D volume.
        """
        cropped_volume = crop_volume(self.image, self.crop_size, self.offsets)
        self.assertEqual(cropped_volume.shape, self.crop_size)

    def test_save_volume_and_mask(self):
        """
        Test saving processed images and masks as .npy files.
        """
        cropped_image = self.image[56:184, 56:184, 13:141]
        cropped_mask = self.mask[56:184, 56:184, 13:141]
        save_volume_and_mask(cropped_image, cropped_mask, self.img_idx, self.save_path)

        # Check that files were saved
        self.assertTrue(os.path.exists(f'{self.save_path}/images/image_0.npy'))
        self.assertTrue(os.path.exists(f'{self.save_path}/masks/mask_0.npy'))

    def test_process_and_save_images(self):
        """
        Test end-to-end processing of images and masks.
        """
        nib.save(nib.Nifti1Image(self.image, np.eye(4)), 'test_image.nii')
        nib.save(nib.Nifti1Image(self.mask, np.eye(4)), 'test_mask.nii')

        image_paths = ['test_image.nii']
        mask_paths = ['test_mask.nii']
        process_and_save_images(image_paths, mask_paths, self.save_path, self.crop_size, self.offsets)

        # Check that files were saved
        self.assertTrue(os.path.exists(f'{self.save_path}/images/image_0.npy'))
        self.assertTrue(os.path.exists(f'{self.save_path}/masks/mask_0.npy'))

        # Clean up test files
        os.remove('test_image.nii')
        os.remove('test_mask.nii')


if __name__ == '__main__':
    unittest.main()
