# files
import glob
import os

# enum
from enum import StrEnum
from io import BytesIO

# scraping
import requests

# image processing
from PIL import Image

# progress bar
from tqdm import tqdm


# create some enums
class Age(StrEnum):
    RANDOM = "none"
    AGE_1_21 = "1-21"
    AGE_21_35 = "22-35"
    AGE_35_50 = "35-50"
    AGE_49_100 = "49-100"


class Race(StrEnum):
    RANDOM = "none"
    ASIAN = "asian"
    WHITE = "white"
    LATINO_HISPANIC = "latino_hispanic"
    MIDDLE_EASTERN = "middle_eastern"
    INDIAN = "indian"
    BLACK = "black"


class Emotion(StrEnum):
    RANDOM = "none"
    HAPPY = "happy"
    NEUTRAL = "neutral"


#
# post request
#
BASE_URL = "https://thispersonnotexist.org"
HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    ),
}
PAYLOAD = {"type": "R", "age": Age.AGE_1_21, "race": Race.RANDOM, "emotion": Emotion.RANDOM}


# path outputs
OUTPUT_DIR = "data/input"
os.makedirs(f"{OUTPUT_DIR}/imgs_0", exist_ok=True)


# get filename
def get_filename(file_name: str, max_files_per_folder: int = 1024) -> str:
    """
    generate a filename for saving images in folders with limited number of files

    Args:
        file_name (str): base file name
        max_files_per_folder (int, optional): max files per folder. Defaults to 1024.
    Returns:
        str: full path filename
    """

    # check existing folders
    n_folders = len(glob.glob(f"{OUTPUT_DIR}/imgs_*")) - 1

    # check amount of files in last folder
    n_images = len(glob.glob(f"{OUTPUT_DIR}/imgs_{n_folders}/**.png"))

    # create new folder if needed
    if n_images >= max_files_per_folder:
        n_folders += 1
        os.makedirs(f"{OUTPUT_DIR}/imgs_{n_folders}", exist_ok=True)

    return f"{OUTPUT_DIR}/imgs_{n_folders}/img_{file_name}.png"


def main():
    """
    Download and process face images from a remote API.
    This function continuously fetches face images from a specified API endpoint,
    processes them by converting to RGB, resizing to 64x64 pixels, and saves them
    to disk. It skips images that have already been downloaded.
    The function performs the following steps:

    1. Loads a set of already downloaded image IDs from the output directory
    2. Makes POST requests to fetch face image IDs from the API
    3. For each new image ID:
        - Downloads the raw image data
        - Converts the image to RGB format
        - Resizes to 64x64 pixels using LANCZOS resampling
        - Saves the processed image to disk
        - Adds the image ID to the tracking set
    4. Skips images that already exist in the output directory
    5. Handles exceptions gracefully and continues processing

    The function runs for 10,000 iterations or until interrupted.

    Note: We download image from "https://thispersonnotexist.org" which is a website that generates images of
    non-existent people using GANs (Generative Adversarial Networks). These images are synthetic and do not correspond
     to real individuals. This is important to consider for ethical and privacy reasons. For more information see README.md.
    """

    # load all images
    image_set = {x.rsplit("/img_", 1)[-1][:-4] for x in glob.glob(f"{OUTPUT_DIR}/imgs_*/**.png")}

    for _ in tqdm(range(10_000)):
        try:
            # make a post request to fetch the image json urls
            response = requests.post(f"{BASE_URL}/load-faces", json=PAYLOAD, headers=HEADERS).json().get("fc")

            # each json contains 8 image ids
            for img_id in response:
                # skip existing images
                if img_id in image_set:
                    print(f"Skipping Image: {img_id}.png")
                    continue

                # get the raw image
                raw = requests.get(f"{BASE_URL}/downloadimage/{img_id}").content

                # Load image from bytes
                img = Image.open(BytesIO(raw))

                # Convert to RGB (handles RGBA, L, P, etc.)
                img = img.convert("RGB")

                # Resize to 64x64
                img = img.resize((64, 64), Image.Resampling.LANCZOS)  # type: ignore

                # Create output directory if it doesn't exist
                img.save(get_filename(img_id))
                image_set.add(img_id)

        except Exception as e:
            print(f"Error occurred: {e}")
            continue


if __name__ == "__main__":
    main()
