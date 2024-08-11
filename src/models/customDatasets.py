from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import math
import random


class MicroStripsDataset(Dataset):
    def create_padded_integer(self, s, N):
        padded_string = s.zfill(N)
        return padded_string

    def image_to_strips_tensor(
        self, image_path, img_channels, sub_img_height, sub_img_width, sub_img_idxs
    ):
        # Calculate number of final images
        final_images_n = len(sub_img_idxs)

        # np array to save images
        sub_images_tensor = torch.zeros(
            (final_images_n, img_channels, sub_img_height, sub_img_width)
        )

        # Load original image
        image = Image.open(image_path)

        # PIL to tensor transformer
        # This transformer will normalize images automatically
        to_tensor = transforms.ToTensor()

        # Get sub images
        for i, img_idx in enumerate(sub_img_idxs):
            # Define the coordinates of the crop box (left, upper, right, lower)
            current_crop_box = (
                0,
                img_idx * sub_img_height,
                sub_img_width,
                (img_idx + 1) * sub_img_height,
            )
            if img_channels == 1:
                sub_images_tensor[i] = to_tensor(
                    image.crop(current_crop_box).convert("L")
                )
            else:
                sub_images_tensor[i] = to_tensor(image.crop(current_crop_box))

        return sub_images_tensor

    def load_data(
        self,
        images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        img_channels,
        sub_img_height,
        sub_img_width,
        sub_img_idxs,
    ):
        # Number of sub images per original image
        n_sub_images = len(sub_img_idxs)

        # The labels csv file must have the following structure: sampleId | sandMineral | siltMineral | clayMineral | sand | silt | clay | om | irdaClass | usdaClass | customClass
        # The first row of the labels csv file should have the columns names and will be ignored
        if label_type == "usda-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -2]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "custom-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -1]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "mineral-regression":
            # The resulting label array structure is: labels[n] = [mineralSand, mineralSilt, mineralClay]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 3)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 1:4]
            labels = torch.zeros(labels_tensor_dim)
        elif label_type == "om-regression":
            # The resulting label array structure is: labels[n] = [sand, silt, clay, om]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 4)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 4:8]
            labels = torch.zeros(labels_tensor_dim)
        else:
            raise Exception("Unknown label type")

        # Create arrays to save training data
        data = torch.zeros(
            (
                len(samples_idxs) * n_sub_images,
                img_channels,
                sub_img_height,
                sub_img_width,
            )
        )

        # Save training data
        for i, img_idx in enumerate(samples_idxs):
            # Log progress
            if i % 100 == 0:
                print(f"{i} samples processed")

            # Save current label
            if (label_type == "usda-classification") or (
                label_type == "custom-classification"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = (
                    labels_info[img_idx] - 1
                )
            elif (label_type == "mineral-regression") or (
                label_type == "om-regression"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = torch.tensor(
                    labels_info[img_idx]
                )

            # Calculate current image path
            current_img_path = (
                images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )
            # Generate current image sub images
            current_sub_images = self.image_to_strips_tensor(
                current_img_path,
                img_channels,
                sub_img_height,
                sub_img_width,
                sub_img_idxs,
            )
            # Save all sub images
            for j, curent_sub_image in enumerate(current_sub_images):
                data[i * n_sub_images + j] = curent_sub_image

        return data, labels

    def __init__(
        self,
        images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        img_channels,
        sub_img_height,
        sub_img_width,
        sub_img_idxs,
        transform=None,
    ):
        self.x, self.y = self.load_data(
            images_dir,
            labels_file_path,
            label_type,
            samples_idxs,
            img_channels,
            sub_img_height,
            sub_img_width,
            sub_img_idxs,
        )

        self.samples_n = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        X = self.x[index]
        y = self.y[index]

        if self.transform:
            X = self.transform(X)

        return X, y

    def __len__(self):
        return self.samples_n


class RotationDataset(Dataset):
    def create_padded_integer(self, s, N):
        padded_string = s.zfill(N)
        return padded_string

    def line(self, m, b, x):
        return m * x + b

    def getCx(self, a, d, theta):
        """
        A small square of dimensions 2d x 2d has to be placed inside a big square of dimensions
        2a x 2a (centerd in the origin) that has been rotated by theta degrees (counterclock wise).
        The small square should be completely inside the rotated big square. This function returns
        the interval the x component of the center of the small square has to live in.

        Parameters:
        a (int):     half the size of one side of the big square
        d (int):     half the size of one side of the small square
        theta (int): angle (in degrees) the big square was rotated

        Returns:
        min, max: min and max values of the interval the x component of the center of the small square
        has to live in.
        """

        if d > a:
            raise Exception("a must be equal or bigger than d")

        angle = theta % 90

        if angle == 0:
            l_min = d - a
            l_max = a - d
        else:
            m1 = math.sin(math.radians(angle)) / math.cos(math.radians(angle))
            m2 = math.sin(math.radians(angle + 90)) / math.cos(math.radians(angle + 90))
            m3 = math.sin(math.radians(angle + 270)) / math.cos(
                math.radians(angle + 270)
            )
            m4 = math.sin(math.radians(angle + 180)) / math.cos(
                math.radians(angle + 180)
            )

            b1 = a / math.cos(math.radians(angle))
            b2 = a / math.cos(math.radians(angle + 90))
            b3 = a / math.cos(math.radians(angle + 270))
            b4 = a / math.cos(math.radians(angle + 180))

            l_min = (2 * d + b2 - b1) / (m1 - m2) + d
            l_max = (2 * d + b4 - b3) / (m3 - m4) - d

        return l_min, l_max

    def getCy(self, a, d, theta, x):
        """
        A small square of dimensions 2d x 2d has to be placed inside a big square of dimensions
        2a x 2a (centerd in the origin) that has been rotated by theta degrees (counterclock wise).
        The small square should be completely inside the rotated big square. Given the x component
        of the center of the small square this function returns the interval the y component of the
        center of the small square has to live in.

        Parameters:
        a (int):     half the size of one side of the big square
        d (int):     half the size of one side of the small square
        theta (int): angle (in degrees) the big square was rotated
        x:           x component of the center of the small square

        Returns:
        min, max: min and max values of the interval the y component of the center of the small square
        has to live in.
        """

        if d > a:
            raise Exception("a must be equal or bigger than d")

        angle = theta % 90

        if angle == 0:
            l_min = d - a
            l_max = a - d
        else:
            m1 = math.sin(math.radians(angle)) / math.cos(math.radians(angle))
            m2 = math.sin(math.radians(angle + 90)) / math.cos(math.radians(angle + 90))
            m3 = math.sin(math.radians(angle + 270)) / math.cos(
                math.radians(angle + 270)
            )
            m4 = math.sin(math.radians(angle + 180)) / math.cos(
                math.radians(angle + 180)
            )

            b1 = a / math.cos(math.radians(angle))
            b2 = a / math.cos(math.radians(angle + 90))
            b3 = a / math.cos(math.radians(angle + 270))
            b4 = a / math.cos(math.radians(angle + 180))

            k_max = min(
                self.line(m1, b1, x - d),
                self.line(m3, b3, x - d),
                self.line(m1, b1, x + d),
                self.line(m3, b3, x + d),
            )
            k_min = max(
                self.line(m2, b2, x - d),
                self.line(m4, b4, x - d),
                self.line(m2, b2, x + d),
                self.line(m4, b4, x + d),
            )

            l_min = k_min + d
            l_max = k_max - d

        return l_min, l_max

    def pToImgCoor(self, x, y, a):
        return x + a, -(y - a)

    def get_sub_images(
        self,
        image_path,
        img_channels,
        img_dim,
        sub_img_dim,
        sub_img_resize_dim,
        rotation_angle,
        intervals_n,
    ):
        # PIL to tensor transformer
        # This transformer will normalize images automatically
        to_tensor = transforms.ToTensor()

        # np array to save images
        sub_images_tensor = torch.zeros(
            (intervals_n, img_channels, sub_img_resize_dim, sub_img_resize_dim)
        )

        # Load original image
        image = Image.open(image_path)

        # Get region of interest
        image = image.crop([0, 0, img_dim, img_dim])

        # Get relative angle of rotation (theta)
        theta = rotation_angle % 90

        # Rotate image
        image = image.rotate(theta)

        # Get needed parameters
        a = int(img_dim / 2)
        d = int(sub_img_dim / 2)

        # Get x-axis limits
        x_min, x_max = self.getCx(a, d, theta)
        x_interval_length = (x_max - x_min) / intervals_n

        # Get sub images
        for i in range(intervals_n):
            # Get limit of current interval
            current_interval_from = int(x_min + i * x_interval_length)
            current_interval_to = int(x_min + (i + 1) * x_interval_length)

            # Get random x from correct interval
            x = random.randint(current_interval_from, current_interval_to)

            # Get y-axis limits based on the value of x
            y_min, y_max = self.getCy(a, d, theta, x)

            # Get random y from correct intervall
            y = random.randint(int(y_min), int(y_max))

            # Define the coordinates of the square's top-left and bottom-right corners
            b_top_left = (x - d, y + d)
            b_bottom_right = (x + d, y - d)

            # Get points in image coordinates
            b_top_left_w, b_top_left_h = self.pToImgCoor(
                b_top_left[0], b_top_left[1], a
            )
            b_bottom_right_w, b_bottom_right_h = self.pToImgCoor(
                b_bottom_right[0], b_bottom_right[1], a
            )

            # Build crop box
            current_crop_box = [
                b_top_left_w,
                b_top_left_h,
                b_bottom_right_w,
                b_bottom_right_h,
            ]

            if img_channels == 1:
                sub_images_tensor[i] = to_tensor(
                    image.crop(current_crop_box)
                    .convert("L")
                    .resize((sub_img_resize_dim, sub_img_resize_dim), Image.NEAREST)
                )
            else:
                sub_images_tensor[i] = to_tensor(
                    image.crop(current_crop_box).resize(
                        (sub_img_resize_dim, sub_img_resize_dim), Image.NEAREST
                    )
                )

        return sub_images_tensor

    def load_data(
        self,
        images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        img_channels,
        img_dim,
        sub_img_dim,
        sub_img_resize_dim,
        rotations,
        intervals_n,
    ):
        # Number of rotations
        rotations_n = len(rotations)

        # Number of sub images per original image
        n_sub_images = rotations_n * intervals_n

        # The labels csv file must have the following structure: sampleId | sandMineral | siltMineral | clayMineral | sand | silt | clay | om | irdaClass | usdaClass | customClass
        # The first row of the labels csv file should have the columns names and will be ignored
        if label_type == "usda-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -2]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "custom-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -1]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "mineral-regression":
            # The resulting label array structure is: labels[n] = [mineralSand, mineralSilt, mineralClay]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 3)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 1:4]
            labels = torch.zeros(labels_tensor_dim)
        elif label_type == "om-regression":
            # The resulting label array structure is: labels[n] = [sand, silt, clay, om]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 4)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 4:8]
            labels = torch.zeros(labels_tensor_dim)
        else:
            raise Exception("Unknown label type")

        # Create arrays to save training data
        data = torch.zeros(
            (
                len(samples_idxs) * n_sub_images,
                img_channels,
                sub_img_resize_dim,
                sub_img_resize_dim,
            )
        )

        # Save training data
        for i, img_idx in enumerate(samples_idxs):
            # Save current label
            if (label_type == "usda-classification") or (
                label_type == "custom-classification"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = (
                    labels_info[img_idx] - 1
                )
            elif (label_type == "mineral-regression") or (
                label_type == "om-regression"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = torch.tensor(
                    labels_info[img_idx]
                )

            # Calculate current image path
            current_img_path = (
                images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )

            for j, current_rotation_angle in enumerate(rotations):
                # Generate current image sub images
                current_sub_images = self.get_sub_images(
                    current_img_path,
                    img_channels,
                    img_dim,
                    sub_img_dim,
                    sub_img_resize_dim,
                    current_rotation_angle,
                    intervals_n,
                )

                # Save all sub images
                for k, curent_sub_image in enumerate(current_sub_images):
                    data[i * n_sub_images + j * intervals_n + k] = curent_sub_image

        return data, labels

    def __init__(
        self,
        images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        img_channels,
        img_dim,
        sub_img_dim,
        sub_img_resize_dim,
        rotations,
        intervals_n,
    ):
        """
        This class will create a Dataset following the next steps for each provided image:
            1.  Crop the original image from the top left corner to get a square image.
            2.  Rotate the croped image
            3.  Based on the rotated image get the valid interval for values of the x component of the center of the sub images based on their size
            4.  The interval in step 4 is divided in N sub intervals and for each sub interval a center for a sub image is selected randomly
            5.  Get 1 sub image for each sub interval
            6.  Steps 2-5 are repeated for each rotation angle specified for each image

        Params:
            images_dir:         path of the dir where the original images are stored
            labels_file_path:   path to the file containing labels information
            label_type:         type of label needed (classification | regression)
            samples_idxs:       array with the samples ids that should be used to build the dataset
            img_channels:       Number of channels to use to build the dataset (3 -> color | 1 -> grayscale)
            img_dim:            Size of the resulting cropped image in step 1
            sub_img_dim:        Size of the obtaiained sub images
            sub_img_resize_dim: Dimension to resize the final sub images
            rotations:          Array with the rotation angles to be applied to each image in step 2
            intervals_n:        Number of sub intervals to use for step 4

        The number of elements in the set is going to be: len(samples_idxs)*len(rotations)*intervals_n
        """

        self.x, self.y = self.load_data(
            images_dir,
            labels_file_path,
            label_type,
            samples_idxs,
            img_channels,
            img_dim,
            sub_img_dim,
            sub_img_resize_dim,
            rotations,
            intervals_n,
        )
        self.samples_n = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.samples_n


class OneImageCropboxRotationDataset(Dataset):
    def create_padded_integer(self, s, N):
        padded_string = s.zfill(N)
        return padded_string

    def get_sub_images(
        self,
        image_path,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        rotations_values,
        flop_values,
        hsv,
    ):
        rotations_n = len(rotations_values)

        # Initialize quadrant tensor
        subimages_tensor = torch.zeros(
            (len(cropboxes) * rotations_n, img_channels, height_resize, width_resize)
        )

        # Load original image
        image = Image.open(image_path)

        # Convert to hsv if needed
        if hsv:
            image = image.convert("HSV")

        # PIL to tensor transformer
        # This transformer will normalize images automatically
        to_tensor = transforms.ToTensor()

        # Get subimages
        for i, current_crop_box in enumerate(cropboxes):
            if img_channels == 1:
                current_subimage = (
                    image.crop(current_crop_box)
                    .convert("L")
                    .resize((height_resize, width_resize), Image.NEAREST)
                )
                for j, (rot, flop) in enumerate(zip(rotations_values, flop_values)):
                    if flop:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                                rot
                            )
                        )
                    else:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.rotate(rot)
                        )

            else:
                current_subimage = image.crop(current_crop_box).resize(
                    (height_resize, width_resize), Image.NEAREST
                )
                for j, (rot, flop) in enumerate(zip(rotations_values, flop_values)):
                    if flop:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                                rot
                            )
                        )
                    else:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.rotate(rot)
                        )

        return subimages_tensor

    def load_data(
        self,
        images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        rotations_values,
        flop_values,
        hsv,
    ):
        n_sub_images = len(cropboxes) * len(rotations_values)

        # The labels csv file must have the following structure: sampleId | sandMineral | siltMineral | clayMineral | sand | silt | clay | om | irdaClass | usdaClass | customClass
        # The first row of the labels csv file should have the columns names and will be ignored
        if label_type == "usda-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -2]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "custom-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -1]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "mineral-regression":
            # The resulting label array structure is: labels[n] = [mineralSand, mineralSilt, mineralClay]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 3)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 1:4]
            labels = torch.zeros(labels_tensor_dim)
        elif label_type == "om-regression":
            # The resulting label array structure is: labels[n] = [sand, silt, clay, om]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 4)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 4:8]
            labels = torch.zeros(labels_tensor_dim)
        else:
            raise Exception("Unknown label type")

        # Create arrays to save training data
        imgs_data = torch.zeros(
            (
                n_sub_images * len(samples_idxs),
                img_channels,
                height_resize,
                width_resize,
            )
        )

        # Save training data
        for i, img_idx in enumerate(samples_idxs):
            # Log progress
            if i % 100 == 0:
                print(f"{i} samples processed")

            # Save current label
            if (label_type == "usda-classification") or (
                label_type == "custom-classification"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = (
                    labels_info[img_idx] - 1
                )
            elif (label_type == "mineral-regression") or (
                label_type == "om-regression"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = torch.tensor(
                    labels_info[img_idx]
                )

            # Calculate current image path
            current_img_path = (
                images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )

            # Generate current image quadrants
            current_subimages = self.get_sub_images(
                current_img_path,
                cropboxes,
                img_channels,
                height_resize,
                width_resize,
                rotations_values,
                flop_values,
                hsv,
            )
            # Save all sub images
            for j in range(n_sub_images):
                imgs_data[i * n_sub_images + j] = current_subimages[j]

        return imgs_data, labels

    def __init__(
        self,
        images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        rotations_values=[0, 90, 180, 270],
        flop_values=[False, True, False, True],
        transform=None,
        hsv=False,
    ):
        self.x, self.y = self.load_data(
            images_dir,
            labels_file_path,
            label_type,
            samples_idxs,
            cropboxes,
            img_channels,
            height_resize,
            width_resize,
            rotations_values,
            flop_values,
            hsv,
        )

        self.samples_n = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        X = self.x[index]
        y = self.y[index]

        if self.transform:
            X = self.transform(X)

        return X, y

    def __len__(self):
        return self.samples_n


class OneImageCropboxRotationDatasetNoLabels(Dataset):
    def create_padded_integer(self, s, N):
        padded_string = s.zfill(N)
        return padded_string

    def get_sub_images(
        self,
        image_path,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        hsv,
    ):
        # Initialize quadrant tensor
        subimages_tensor = torch.zeros(
            (len(cropboxes), img_channels, height_resize, width_resize)
        )

        # Load original image
        image = Image.open(image_path)

        # Convert to hsv if needed
        if hsv:
            image = image.convert("HSV")

        # PIL to tensor transformer
        # This transformer will normalize images automatically
        to_tensor = transforms.ToTensor()

        # Get subimages
        for i, current_crop_box in enumerate(cropboxes):
            if img_channels == 1:
                subimages_tensor[i] = to_tensor(
                    image.crop(current_crop_box)
                    .convert("L")
                    .resize((height_resize, width_resize), Image.NEAREST)
                )
            else:
                subimages_tensor[i] = to_tensor(
                    image.crop(current_crop_box).resize(
                        (height_resize, width_resize), Image.NEAREST
                    )
                )

        return subimages_tensor

    def load_data(
        self,
        images_dir,
        samples_idxs,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        hsv,
    ):
        n_sub_images = len(cropboxes)

        # Create arrays to save training data
        imgs_data = torch.zeros(
            (
                n_sub_images * len(samples_idxs),
                img_channels,
                height_resize,
                width_resize,
            )
        )

        # Save training data
        for i, img_idx in enumerate(samples_idxs):
            # Log progress
            if i % 100 == 0:
                print(f"{i} samples processed")

            # Calculate current image path
            current_img_path = (
                images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )

            # Generate current image quadrants
            current_subimages = self.get_sub_images(
                current_img_path,
                cropboxes,
                img_channels,
                height_resize,
                width_resize,
                hsv,
            )
            # Save all sub images
            for j in range(n_sub_images):
                imgs_data[i * n_sub_images + j] = current_subimages[j]

        return imgs_data

    def __init__(
        self,
        images_dir,
        samples_idxs,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        transform=None,
        hsv=False,
    ):
        self.x = self.load_data(
            images_dir,
            samples_idxs,
            cropboxes,
            img_channels,
            height_resize,
            width_resize,
            hsv,
        )

        self.samples_n = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        X = self.x[index]

        if self.transform:
            X = self.transform(X)

        return X

    def __len__(self):
        return self.samples_n


class TwoImagesCropboxRotationDataset(Dataset):
    def create_padded_integer(self, s, N):
        padded_string = s.zfill(N)
        return padded_string

    def get_sub_images(
        self,
        image_path,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        rotations_values,
        flop_values,
        hsv,
    ):
        rotations_n = len(rotations_values)

        # Initialize quadrant tensor
        subimages_tensor = torch.zeros(
            (len(cropboxes) * rotations_n, img_channels, height_resize, width_resize)
        )

        # Load original image
        image = Image.open(image_path)

        # Convert to hsv if needed
        if hsv:
            image = image.convert("HSV")

        # PIL to tensor transformer
        # This transformer will normalize images automatically
        to_tensor = transforms.ToTensor()

        # Get subimages
        for i, current_crop_box in enumerate(cropboxes):
            if img_channels == 1:
                current_subimage = (
                    image.crop(current_crop_box)
                    .convert("L")
                    .resize((height_resize, width_resize), Image.NEAREST)
                )
                for j, (rot, flop) in enumerate(zip(rotations_values, flop_values)):
                    if flop:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                                rot
                            )
                        )
                    else:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.rotate(rot)
                        )

            else:
                current_subimage = image.crop(current_crop_box).resize(
                    (height_resize, width_resize), Image.NEAREST
                )
                for j, (rot, flop) in enumerate(zip(rotations_values, flop_values)):
                    if flop:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.transpose(Image.FLIP_LEFT_RIGHT).rotate(
                                rot
                            )
                        )
                    else:
                        subimages_tensor[i * rotations_n + j] = to_tensor(
                            current_subimage.rotate(rot)
                        )

        return subimages_tensor

    def load_data(
        self,
        camera_images_dir,
        micro_images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        camera_cropboxes,
        micro_cropboxes,
        camera_img_channels,
        micro_img_channels,
        camera_height_resize,
        camera_width_resize,
        micro_height_resize,
        micro_width_resize,
        rotations_values,
        flop_values,
        hsv,
    ):
        if len(camera_cropboxes) != len(micro_cropboxes):
            raise Exception("Number of cropboxes must be the same for camera and micro")
        n_sub_images = len(micro_cropboxes) * len(rotations_values)

        # The labels csv file must have the following structure: sampleId | sandMineral | siltMineral | clayMineral | sand | silt | clay | om | irdaClass | usdaClass | customClass
        # The first row of the labels csv file should have the columns names and will be ignored
        if label_type == "usda-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -2]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "custom-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -1]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "mineral-regression":
            # The resulting label array structure is: labels[n] = [mineralSand, mineralSilt, mineralClay]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 3)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 1:4]
            labels = torch.zeros(labels_tensor_dim)
        elif label_type == "om-regression":
            # The resulting label array structure is: labels[n] = [sand, silt, clay, om]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 4)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 4:8]
            labels = torch.zeros(labels_tensor_dim)
        else:
            raise Exception("Unknown label type")

        # Create arrays to save training data
        camera_data = torch.zeros(
            (
                n_sub_images * len(samples_idxs),
                camera_img_channels,
                camera_height_resize,
                camera_width_resize,
            )
        )
        micro_data = torch.zeros(
            (
                n_sub_images * len(samples_idxs),
                micro_img_channels,
                micro_height_resize,
                micro_width_resize,
            )
        )

        # Save training data
        for i, img_idx in enumerate(samples_idxs):
            # Log progress
            if i % 100 == 0:
                print(f"{i} samples processed")

            # Save current label
            if (label_type == "usda-classification") or (
                label_type == "custom-classification"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = (
                    labels_info[img_idx] - 1
                )
            elif (label_type == "mineral-regression") or (
                label_type == "om-regression"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = torch.tensor(
                    labels_info[img_idx]
                )

            # Calculate current image path
            camera_current_img_path = (
                camera_images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )
            micro_current_img_path = (
                micro_images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )

            # Generate current image quadrants
            camera_current_subimages = self.get_sub_images(
                camera_current_img_path,
                camera_cropboxes,
                camera_img_channels,
                camera_height_resize,
                camera_width_resize,
                rotations_values,
                flop_values,
                hsv,
            )
            micro_current_subimages = self.get_sub_images(
                micro_current_img_path,
                micro_cropboxes,
                micro_img_channels,
                micro_height_resize,
                micro_width_resize,
                rotations_values,
                flop_values,
                hsv,
            )

            # Save all sub images
            for j in range(n_sub_images):
                camera_data[i * n_sub_images + j] = camera_current_subimages[j]
                micro_data[i * n_sub_images + j] = micro_current_subimages[j]

        return camera_data, micro_data, labels

    def __init__(
        self,
        camera_images_dir,
        micro_images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        camera_cropboxes,
        micro_cropboxes,
        camera_img_channels,
        micro_img_channels,
        camera_height_resize,
        camera_width_resize,
        micro_height_resize,
        micro_width_resize,
        rotations_values=[0, 90, 180, 270],
        flop_values=[False, True, False, True],
        camera_transform=None,
        micro_transform=None,
        hsv=False,
    ):
        self.x_camera, self.x_micro, self.y = self.load_data(
            camera_images_dir,
            micro_images_dir,
            labels_file_path,
            label_type,
            samples_idxs,
            camera_cropboxes,
            micro_cropboxes,
            camera_img_channels,
            micro_img_channels,
            camera_height_resize,
            camera_width_resize,
            micro_height_resize,
            micro_width_resize,
            rotations_values,
            flop_values,
            hsv,
        )

        self.samples_n = self.x_camera.shape[0]
        self.camera_transform = camera_transform
        self.micro_transform = micro_transform

    def __getitem__(self, index):
        x_camera = self.x_camera[index]
        x_micro = self.x_micro[index]
        y = self.y[index]

        if self.camera_transform:
            x_camera = self.camera_transform(x_camera)

        if self.micro_transform:
            x_micro = self.micro_transform(x_micro)

        return x_camera, x_micro, y

    def __len__(self):
        return self.samples_n


class TwoImagesCropboxRotationDatasetNoLabels(Dataset):
    def create_padded_integer(self, s, N):
        padded_string = s.zfill(N)
        return padded_string

    def get_sub_images(
        self,
        image_path,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        hsv,
    ):
        # Initialize quadrant tensor
        subimages_tensor = torch.zeros(
            (len(cropboxes), img_channels, height_resize, width_resize)
        )

        # Load original image
        image = Image.open(image_path)

        # Convert to hsv if needed
        if hsv:
            image = image.convert("HSV")

        # PIL to tensor transformer
        # This transformer will normalize images automatically
        to_tensor = transforms.ToTensor()

        # Get subimages
        for i, current_crop_box in enumerate(cropboxes):
            if img_channels == 1:
                subimages_tensor[i] = to_tensor(
                    image.crop(current_crop_box)
                    .convert("L")
                    .resize((height_resize, width_resize), Image.NEAREST)
                )

            else:
                subimages_tensor[i] = to_tensor(
                    image.crop(current_crop_box).resize(
                        (height_resize, width_resize), Image.NEAREST
                    )
                )

        return subimages_tensor

    def load_data(
        self,
        camera_images_dir,
        micro_images_dir,
        samples_idxs,
        camera_cropboxes,
        micro_cropboxes,
        camera_img_channels,
        micro_img_channels,
        camera_height_resize,
        camera_width_resize,
        micro_height_resize,
        micro_width_resize,
        hsv,
    ):
        if len(camera_cropboxes) != len(micro_cropboxes):
            raise Exception("Number of cropboxes must be the same for camera and micro")
        n_sub_images = len(micro_cropboxes)

        # Create arrays to save training data
        camera_data = torch.zeros(
            (
                n_sub_images * len(samples_idxs),
                camera_img_channels,
                camera_height_resize,
                camera_width_resize,
            )
        )
        micro_data = torch.zeros(
            (
                n_sub_images * len(samples_idxs),
                micro_img_channels,
                micro_height_resize,
                micro_width_resize,
            )
        )

        # Save training data
        for i, img_idx in enumerate(samples_idxs):
            # Log progress
            if i % 100 == 0:
                print(f"{i} samples processed")

            # Calculate current image path
            camera_current_img_path = (
                camera_images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )
            micro_current_img_path = (
                micro_images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )

            # Generate current image quadrants
            camera_current_subimages = self.get_sub_images(
                camera_current_img_path,
                camera_cropboxes,
                camera_img_channels,
                camera_height_resize,
                camera_width_resize,
                hsv,
            )
            micro_current_subimages = self.get_sub_images(
                micro_current_img_path,
                micro_cropboxes,
                micro_img_channels,
                micro_height_resize,
                micro_width_resize,
                hsv,
            )

            # Save all sub images
            for j in range(n_sub_images):
                camera_data[i * n_sub_images + j] = camera_current_subimages[j]
                micro_data[i * n_sub_images + j] = micro_current_subimages[j]

        return camera_data, micro_data

    def __init__(
        self,
        camera_images_dir,
        micro_images_dir,
        samples_idxs,
        camera_cropboxes,
        micro_cropboxes,
        camera_img_channels,
        micro_img_channels,
        camera_height_resize,
        camera_width_resize,
        micro_height_resize,
        micro_width_resize,
        camera_transform=None,
        micro_transform=None,
        hsv=False,
    ):
        self.x_camera, self.x_micro = self.load_data(
            camera_images_dir,
            micro_images_dir,
            samples_idxs,
            camera_cropboxes,
            micro_cropboxes,
            camera_img_channels,
            micro_img_channels,
            camera_height_resize,
            camera_width_resize,
            micro_height_resize,
            micro_width_resize,
            hsv,
        )

        self.samples_n = self.x_camera.shape[0]
        self.camera_transform = camera_transform
        self.micro_transform = micro_transform

    def __getitem__(self, index):
        x_camera = self.x_camera[index]
        x_micro = self.x_micro[index]

        if self.camera_transform:
            x_camera = self.camera_transform(x_camera)

        if self.micro_transform:
            x_micro = self.micro_transform(x_micro)

        return x_camera, x_micro

    def __len__(self):
        return self.samples_n


class TwoImagesMergeCropboxRotationDataset(Dataset):
    def create_padded_integer(self, s, N):
        padded_string = s.zfill(N)
        return padded_string

    def get_sub_images(
        self,
        image_path,
        cropboxes,
        img_channels,
        height_resize,
        width_resize,
        rotations_values,
    ):
        rotations_n = len(rotations_values)

        # Initialize quadrant tensor
        subimages_tensor = torch.zeros(
            (len(cropboxes) * rotations_n, img_channels, height_resize, width_resize)
        )

        # Load original image
        image = Image.open(image_path)

        # PIL to tensor transformer
        # This transformer will normalize images automatically
        to_tensor = transforms.ToTensor()

        # Get subimages
        for i, current_crop_box in enumerate(cropboxes):
            if img_channels == 1:
                current_subimage = (
                    image.crop(current_crop_box)
                    .convert("L")
                    .resize((height_resize, width_resize), Image.NEAREST)
                )
                for j, rot in enumerate(rotations_values):
                    subimages_tensor[i * rotations_n + j] = to_tensor(
                        current_subimage.rotate(rot)
                    )

            else:
                current_subimage = image.crop(current_crop_box).resize(
                    (height_resize, width_resize), Image.NEAREST
                )
                for j, rot in enumerate(rotations_values):
                    subimages_tensor[i * rotations_n + j] = to_tensor(
                        current_subimage.rotate(rot)
                    )

        return subimages_tensor

    def load_data(
        self,
        camera_images_dir,
        micro_images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        camera_cropboxes,
        micro_cropboxes,
        img_channels,
        height_resize,
        width_resize,
        rotations_values,
    ):
        if len(camera_cropboxes) != len(micro_cropboxes):
            raise Exception("Number of cropboxes must be the same for camera and micro")
        n_sub_images = len(micro_cropboxes) * len(rotations_values)

        # The labels csv file must have the following structure: sampleId | sandMineral | siltMineral | clayMineral | sand | silt | clay | om | irdaClass | usdaClass | customClass
        # The first row of the labels csv file should have the columns names and will be ignored
        if label_type == "usda-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -2]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "custom-classification":
            # The resulting label array structure is: labels[n] = class
            labels_tensor_dim = len(samples_idxs) * n_sub_images
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.int32, skiprows=1
            )[:, -1]
            labels = torch.zeros(labels_tensor_dim, dtype=torch.long)
        elif label_type == "mineral-regression":
            # The resulting label array structure is: labels[n] = [mineralSand, mineralSilt, mineralClay]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 3)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 1:4]
            labels = torch.zeros(labels_tensor_dim)
        elif label_type == "om-regression":
            # The resulting label array structure is: labels[n] = [sand, silt, clay, om]
            labels_tensor_dim = (len(samples_idxs) * n_sub_images, 4)
            labels_info = np.loadtxt(
                labels_file_path, delimiter=",", dtype=np.float32, skiprows=1
            )[:, 4:8]
            labels = torch.zeros(labels_tensor_dim)
        else:
            raise Exception("Unknown label type")

        # Create arrays to save training data
        data = torch.zeros(
            (
                n_sub_images * len(samples_idxs),
                img_channels * 2,
                height_resize,
                width_resize,
            )
        )

        # Save training data
        for i, img_idx in enumerate(samples_idxs):
            # Log progress
            if i % 100 == 0:
                print(f"{i} samples processed")

            # Save current label
            if (label_type == "usda-classification") or (
                label_type == "custom-classification"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = (
                    labels_info[img_idx] - 1
                )
            elif (label_type == "mineral-regression") or (
                label_type == "om-regression"
            ):
                labels[i * n_sub_images : (i + 1) * n_sub_images] = torch.tensor(
                    labels_info[img_idx]
                )

            # Calculate current image path
            camera_current_img_path = (
                camera_images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )
            micro_current_img_path = (
                micro_images_dir
                + "/"
                + self.create_padded_integer(str(img_idx + 1), 3)
                + ".jpg"
            )

            # Generate current image quadrants
            camera_current_subimages = self.get_sub_images(
                camera_current_img_path,
                camera_cropboxes,
                img_channels,
                height_resize,
                width_resize,
                rotations_values,
            )
            micro_current_subimages = self.get_sub_images(
                micro_current_img_path,
                micro_cropboxes,
                img_channels,
                height_resize,
                width_resize,
                rotations_values,
            )

            # Save all sub images
            for j in range(n_sub_images):
                data[i * n_sub_images + j] = torch.cat(
                    [camera_current_subimages[j], micro_current_subimages[j]], dim=0
                )

        return data, labels

    def __init__(
        self,
        camera_images_dir,
        micro_images_dir,
        labels_file_path,
        label_type,
        samples_idxs,
        camera_cropboxes,
        micro_cropboxes,
        img_channels,
        height_resize,
        width_resize,
        rotations_values=[0, 90, 180, 270],
        transform=None,
    ):
        self.x, self.y = self.load_data(
            camera_images_dir,
            micro_images_dir,
            labels_file_path,
            label_type,
            samples_idxs,
            camera_cropboxes,
            micro_cropboxes,
            img_channels,
            height_resize,
            width_resize,
            rotations_values,
        )

        self.samples_n = self.x.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.samples_n


##################### Helper functions #####################


def createCropBoxes(N, sub_img_dim):
    """
    This function creates NxN cropboxes of size sub_img_dim. The cropboxes are built by dividing a square in the X and Y directions
    by increments of sub_img_dim.
    """
    crop_boxes = []

    # Define the coordinates of the crop box (left, upper, right, lower)
    for i in range(N):
        current_left = i * sub_img_dim
        current_right = (i + 1) * sub_img_dim
        for j in range(N):
            current_top = j * sub_img_dim
            current_bottom = (j + 1) * sub_img_dim
            crop_boxes.append(
                (current_left, current_top, current_right, current_bottom)
            )

    return crop_boxes
