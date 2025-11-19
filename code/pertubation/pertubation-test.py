from pathlib import Path
import cv2
import numpy as np

IMAGE_PATH = "/home/mozi/datasets/visdrone/VisDrone2019-VID-test-dev/sequences/uav0000009_03358_v/0000001.jpg"

def blur_image(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def noise_image(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_img = cv2.add(image, gauss)
    return noisy_img

def motion_blur_image(image, kernel_size=15):
    # Create a motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    # Apply the kernel to the image
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def vibration_blur_image(image, max_shift=5):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, np.random.randint(-max_shift, max_shift)],
                    [0, 1, np.random.randint(-max_shift, max_shift)]])
    blurred = cv2.warpAffine(image, M, (cols, rows))
    return blurred

if __name__ == "__main__":
    if Path(IMAGE_PATH).exists() is False:
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
    img = cv2.imread(IMAGE_PATH)
    blur_3 = blur_image(img, 3)
    blur_7 = blur_image(img, 7)
    blur_15 = blur_image(img, 15)

    noise_2 = noise_image(img, sigma=10)
    noise_5 = noise_image(img, sigma=20)
    noise_15 = noise_image(img, sigma=50)
    noise_25 = noise_image(img, sigma=100)

    noise_mean_0 = noise_image(img, mean=0, sigma=15)
    noise_mean_10 = noise_image(img, mean=20, sigma=15)
    noise_mean_neg10 = noise_image(img, mean=30, sigma=15)
    noise_image_50 = noise_image(img, mean=50, sigma=15)

    motion_blur_5 = motion_blur_image(img, kernel_size=5)
    motion_blur_15 = motion_blur_image(img, kernel_size=15)
    motion_blur_25 = motion_blur_image(img, kernel_size=25)
    motion_blur_35 = motion_blur_image(img, kernel_size=35)

# Stack horizontally
    h = 300
    blurred_images = [
        cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(blur_3, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(blur_7, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(blur_15, (int(img.shape[1] * h / img.shape[0]), h))
    ]

    # noised_images = [
    #     cv2.resize(noise_2, (int(img.shape[1] * h / img.shape[0]), h)),
    #     cv2.resize(noise_5, (int(img.shape[1] * h / img.shape[0]), h)),
    #     cv2.resize(noise_15, (int(img.shape[1] * h / img.shape[0]), h)),
    #     cv2.resize(noise_25, (int(img.shape[1] * h / img.shape[0]), h))
    # ]

    noised_mean_images = [
        cv2.resize(noise_mean_0, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(noise_mean_10, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(noise_mean_neg10, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(noise_image_50, (int(img.shape[1] * h / img.shape[0]), h))
    ]

    motion_blur_images = [
        cv2.resize(motion_blur_5, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(motion_blur_15, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(motion_blur_25, (int(img.shape[1] * h / img.shape[0]), h)),
        cv2.resize(motion_blur_35, (int(img.shape[1] * h / img.shape[0]), h))
    ]

    vibration_blur_images = [
        cv2.resize(vibration_blur_image(img,i*4+1), (int(img.shape[1] * h / img.shape[0]), h)) for i in range(4)
    ]

    blurred_images = np.hstack(blurred_images)
    # noise_combined = np.hstack(noised_images)
    noised_mean_combined = np.hstack(noised_mean_images)
    motion_blur_combined = np.hstack(motion_blur_images)
    vibration_blur_combined = np.hstack(vibration_blur_images)
    combined = np.vstack([blurred_images, noised_mean_combined, motion_blur_combined, vibration_blur_combined])
    cv2.imshow("Image", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()