import picamera
import time
import csv
import os
import cv2


def printSettings(camera):
    finalSettings = {
        "iso": camera.iso,
        "exposure_speed": camera.exposure_speed,
        "awb_gains": camera.awb_gains,
        "framerate": camera.framerate,
        "exposure_compensation": camera.exposure_compensation,
    }
    print(finalSettings)


def getCameraConfiguration(camera):
    config = {
        "iso": camera.iso,
        "exposure_speed": camera.exposure_speed,
        "analog_gain": camera.analog_gain,
        "digital_gain": camera.digital_gain,
        "awb_gains": camera.awb_gains,
        "framerate": camera.framerate,
        "exposure_compensation": camera.exposure_compensation,
    }
    return config


def microscope_image(data_dir, img_name):

    # video capture source camera (Here webcam of laptop)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

    for x in range(10):
        time.sleep(0.5)
        ret, frame = cap.read()

    cap.release()
    if ret:
        cv2.imwrite(f"{data_dir}/Images/{img_name}_micro.jpg", frame)


def main():
    # Ask for directory where data is going to be saved
    data_dir = input("Enter full path of data directory: ")
    csv_name = input("Enter name of csv file: ")

    # Create dir to save images
    imgs_dir = f"{data_dir}/Images"
    if not os.path.exists(imgs_dir):
        os.mkdir(imgs_dir)

    with picamera.PiCamera() as camera:
        # Set camera settings
        camera.resolution = (2592, 1944)

        # Camera warm-up time
        time.sleep(3)

        # Get camera config names
        cam_configs_names = list(getCameraConfiguration(camera).keys())

        with open(f"{data_dir}/{csv_name}.csv", "w", newline="") as csvFile:
            writer = csv.writer(csvFile)

            # write the camera configuration anmes as the first row of the CSV file
            writer.writerow(["name"] + cam_configs_names)

            while True:
                # Ask for image name
                img_name = input("Enter image name: ")

                # Break loop
                if img_name == "exit":
                    break

                # Print camera settings
                camera.capture(f"{data_dir}/Images/{img_name}.jpg", "jpeg")
                current_camera_configs = getCameraConfiguration(camera)

                # Save picture metadata
                row = [img_name]
                for key in cam_configs_names:
                    row.append(current_camera_configs[key])
                writer.writerow(row)

                microscope_image(data_dir, img_name)


main()
