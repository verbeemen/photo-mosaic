import glob

import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from tqdm import tqdm

# --- Configuration ---
INPUT = "data/input/imgs_*/**.png"
# INPUT_TARGET_PATH = "data/input/cat.png"
INPUT_TARGET_PATH = "data/input/priest.jpeg"
OUTPUT_DIR = "data/output"
OUTPUT_NAME = "innosaint_mosaic_structural_brick.png"

FEATURE_SIZE = 4  # Comparison window (4x4)
N_TILES_H = 128  # Number of tiles across the width
GHOSTING_ALPHA = 0.25  # 0.0 = original tiles, 1.0 = target image colors


def main() -> None:
    """
    Main function to create a photo mosaic from a target image and a library of tile images.
    This function performs the following steps:
    1. Loads and resizes the target image to work dimensions based on N_TILES_H and FEATURE_SIZE
    2. Indexes all library images, creating both full-resolution tiles and downsampled feature vectors
    3. Builds a KD-Tree for efficient nearest-neighbor matching of image features
    4. Creates an output canvas with extra space to accommodate brick-pattern offsets
    5. Constructs the mosaic by:
        - Iterating through grid positions with alternating row offsets (brick pattern)
        - Matching each target block to the best-fitting library tile using feature vectors
        - Optionally tinting tiles with target colors based on GHOSTING_ALPHA
        - Placing tiles on the output canvas
    6. Crops and saves the final mosaic image
    Returns:
         None: Saves the mosaic to disk at OUTPUT_DIR/OUTPUT_NAME
    """
    #
    # 1. Load the target image and determine dimensions
    #
    target_img_raw = Image.open(INPUT_TARGET_PATH).convert("RGB")

    # We want N_TILES_H across. Each tile represents a FEATURE_SIZE block.
    # So the target "work image" should be (N_TILES_H * FEATURE_SIZE) wide.
    aspect_ratio = target_img_raw.height / target_img_raw.width
    work_w = N_TILES_H * FEATURE_SIZE
    work_h = int(work_w * aspect_ratio)

    # Ensure work_h is a multiple of FEATURE_SIZE to avoid partial blocks
    work_h = (work_h // FEATURE_SIZE) * FEATURE_SIZE

    # Create the target work image
    target_work_img = np.array(target_img_raw.resize((work_w, work_h), Image.Resampling.LANCZOS))

    #
    # 2. Load and Pre-process Library Images
    #
    # Collect the image paths
    images_paths = glob.glob(INPUT)

    # Preload images and get their tile size - assuming they are all the same size
    first_tile = Image.open(images_paths[0])
    t_width, t_height = first_tile.size  # Original tile resolution (e.g., 64x64)

    # pre-allocate the working arrays
    # - The images array holds the actual image data
    # - The images_features array holds the downsampled feature vectors for matching
    images = np.zeros((len(images_paths), t_height, t_width, 3), dtype=np.uint8)
    images_features = np.zeros((len(images_paths), FEATURE_SIZE * FEATURE_SIZE * 3), dtype=np.float32)

    # Load the images
    print(f"Indexing {len(images_paths)} library images...")
    for i, path in enumerate(tqdm(images_paths)):
        with Image.open(path).convert("RGB") as img:
            # Save the "tile" image
            images[i] = np.array(img)

            # Save the 4x4x3 feature vector as a flattened array
            images_features[i] = np.array(img.resize((FEATURE_SIZE, FEATURE_SIZE), Image.Resampling.LANCZOS)).flatten()

    # Store the images features in a KD-Tree for fast nearest-neighbor lookup
    image_tree = KDTree(images_features)
    del images_features  # Free memory

    #
    # 3. Setup Output Canvas (with extra space for the brick offset)
    #

    # Pre-calculate grid size and canvas size
    grid_rows = work_h // FEATURE_SIZE
    grid_cols = work_w // FEATURE_SIZE

    offset_pixels = t_width // 2
    canvas_w = (grid_cols * t_width) + offset_pixels
    canvas_h = grid_rows * t_height

    # Pre-allocate output canvas
    output_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    #
    # 4. Build the Mosaic
    #
    print("Building mosaic with structural matching and brick offsets...")
    for r in tqdm(range(grid_rows)):
        # Calculate brick offset for this row
        is_odd = r % 2 == 1
        row_x_offset = offset_pixels if is_odd else 0

        # Sample the target slightly to the right for odd rows to match the shift
        feat_x_offset = (FEATURE_SIZE // 2) if is_odd else 0

        # Loop over columns
        for c in range(grid_cols):
            # Coordinates in the target work image
            target_y = r * FEATURE_SIZE
            target_x = c * FEATURE_SIZE + feat_x_offset

            # Boundary check for sampling
            target_x = min(target_x, work_w - FEATURE_SIZE)

            # Extract feature vector from target
            target_sub = target_work_img[
                target_y : target_y + FEATURE_SIZE, target_x : target_x + FEATURE_SIZE, :
            ].flatten()

            # Find best match (pick from top 7 for variety)
            _, indices = image_tree.query(target_sub, k=7)
            best_id = np.random.choice(indices)

            # Get the tile image
            tile = images[best_id].copy().astype(np.float32)

            # Optional: Tint the tile with the average color of the target block
            avg_color = target_sub.reshape(-1, 3).mean(axis=0)
            tile = (tile * (1 - GHOSTING_ALPHA) + avg_color * GHOSTING_ALPHA).astype(np.uint8)

            # Paste into canvas
            out_y = r * t_height
            out_x = c * t_width + row_x_offset
            output_canvas[out_y : out_y + t_height, out_x : out_x + t_width] = tile

    #
    # 5. Save result (Crop the extra offset width for a clean edge)
    #
    final_image = Image.fromarray(output_canvas).crop((offset_pixels, 0, canvas_w - offset_pixels, canvas_h))
    final_image.save(f"{OUTPUT_DIR}/{OUTPUT_NAME}")
    print(f"Done! Saved to {OUTPUT_DIR}/{OUTPUT_NAME}")


if __name__ == "__main__":
    main()
