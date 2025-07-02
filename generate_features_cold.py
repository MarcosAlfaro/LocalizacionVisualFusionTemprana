# Script to compute all feature maps simultaneously for COLD dataset:
# - Grayscale
# - Hue
# - Gradient Magnitude
# - Gradient Orientation
# All features are resized to 384x192 pixels and saved as .npy files

import os
import cv2
import numpy as np

# Source and destination base paths for COLD dataset
SOURCE_BASE_PATH = "/home/arvc/Marcos/INVESTIGACION/0_DATASETS/COLD/"
DEST_BASE_PATH = f"{SOURCE_BASE_PATH}FEATURES_PRUEBA/"

# COLD environments to process
ENVIRONMENTS = ["FR_A", "FR_B", "SA_A", "SA_B"]

def rgb_to_hue(rgb_image):
    """Convert RGB image to hue channel using OpenCV (vectorized version)."""
    # Convert RGB to HSV using OpenCV's efficient implementation
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Extract hue channel and normalize to [0,1] range
    # OpenCV uses range [0,179] for hue
    hue_channel = hsv_image[:,:,0].astype(np.float32) / 179.0
    
    return hue_channel

def compute_all_features(image_path):
    """
    Compute all feature maps from a single image:
    - Grayscale
    - Hue
    - Gradient Magnitude
    - Gradient Orientation (in degrees 0-360)

    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, None, None, None
        
        # Convert BGR to RGB for hue computation
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Generate intensity image
        intensity = np.mean(image_rgb, axis=2).astype(np.float32)
        
        # 2. Generate hue image
        hue = rgb_to_hue(image_rgb)
        
        # 3. & 4. Compute gradient features using intensity
        # Compute gradients using Sobel
        grad_x = cv2.Sobel(intensity, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(intensity, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)
        
        # Compute gradient orientation in degrees (0-360)
        # arctan2 returns angle in radians in range (-π, π)
        orientation = np.arctan2(grad_y, grad_x)
        
        # Convert to degrees and shift from [-180, 180] to [0, 360]
        orientation = np.degrees(orientation)
        orientation = (orientation + 360) % 360
        orientation = orientation.astype(np.float32)
        
        return intensity, hue, magnitude, orientation
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None, None
    finally:
        # Explicitly clean up
        if 'image' in locals():
            del image
        if 'image_rgb' in locals():
            del image_rgb
        if 'grad_x' in locals():
            del grad_x
        if 'grad_y' in locals():
            del grad_y

def process_environment_images(env):
    """Process images for a given COLD environment."""
    env_path = os.path.join(SOURCE_BASE_PATH, env)
    
    # Check if environment directory exists
    if not os.path.exists(env_path):
        print(f"Environment path {env_path} does not exist. Skipping.")
        return

    print(f"Processing images for COLD environment: {env}")
    
    # COLD dataset structure: each environment contains image files directly
    # or in subdirectories. Let's check both cases.
    

    # Check for subdirectories containing images
    subdirs = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
    
    for subdir in subdirs:
        # Skip if subdir is not a valid environment subdirectory
        if subdir not in ["Train", "Validation", "TestCloudy", "TestNight", "TestSunny"]:
            print(f"Skipping unknown subdirectory: {subdir}")
            continue

        subdir_path = os.path.join(env_path, subdir)
        rooms = os.listdir(subdir_path)

        for room in rooms:
            room_path = os.path.join(subdir_path, room)

            # Look for images in this subdirectory
            images = [f for f in os.listdir(room_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]
            
            if images:
                print(f"  Found {len(images)} images in {env}/{subdir}/{room} folder")
                process_image_list(room_path, images, env, subdir, room)

def process_image_list(image_dir, image_files, env, subdir_name, room):
    """Process a list of images from a specific directory."""
    
    # Create destination directories for all features
    base_dest_path = DEST_BASE_PATH
    
    intensity_dest_dir = os.path.join(base_dest_path, "INTENSITY", env, subdir_name, room)
    hue_dest_dir = os.path.join(base_dest_path, "HUE", env, subdir_name, room)
    magnitude_dest_dir = os.path.join(base_dest_path, "MAGNITUDE", env, subdir_name, room)
    angle_dest_dir = os.path.join(base_dest_path, "ANGLE", env, subdir_name, room)
    
    os.makedirs(intensity_dest_dir, exist_ok=True)
    os.makedirs(hue_dest_dir, exist_ok=True)
    os.makedirs(magnitude_dest_dir, exist_ok=True)
    os.makedirs(angle_dest_dir, exist_ok=True)
    
    # Process all images
    for i, image_file in enumerate(image_files):
        source_image_path = os.path.join(image_dir, image_file)
        
        # Process image to extract all features
        location_str = f"{env}/{subdir_name}" if subdir_name != "direct" else env
        print(f"[{i+1}/{len(image_files)}] Processing {location_str} image: {image_file}")
        
        intensity, hue, magnitude, orientation = compute_all_features(source_image_path)
        
        if intensity is not None and hue is not None and magnitude is not None and orientation is not None:
            # Save all features with the same base filename
            base_filename = os.path.splitext(image_file)[0]
            
            # Save intensity
            intensity_dest = os.path.join(intensity_dest_dir, base_filename + ".npy")
            np.save(intensity_dest, intensity)
            
            # Save hue
            hue_dest = os.path.join(hue_dest_dir, base_filename + ".npy")
            np.save(hue_dest, hue)
            
            # Save gradient magnitude
            magnitude_dest = os.path.join(magnitude_dest_dir, base_filename + ".npy")
            np.save(magnitude_dest, magnitude)
            
            # Save gradient orientation
            angle_dest = os.path.join(angle_dest_dir, base_filename + ".npy")
            np.save(angle_dest, orientation)
        

def main():
    """Main function to process all images in the COLD dataset environments."""
    print("Starting comprehensive feature extraction for COLD dataset...")
    print("Features to be generated:")
    print("  - Intensity (INTENSITY/)")
    print("  - Hue (HUE/)")
    print("  - Gradient Magnitude (MAGNITUDE/)")
    print("  - Gradient Orientation (ANGLE/)")
    print()
    
    # Check if COLD dataset base path exists
    if not os.path.exists(SOURCE_BASE_PATH):
        print(f"Error: COLD dataset path does not exist: {SOURCE_BASE_PATH}")
        print("Please update SOURCE_BASE_PATH in the script to point to your COLD dataset location.")
        return
    
    # Process each environment
    for env in ENVIRONMENTS:
        print(f"\n==== Processing COLD environment: {env} ====")
        
        # Process images in this environment
        process_environment_images(env)
    
    print("\nCOLD dataset comprehensive feature extraction complete!")
    print("All four feature maps have been generated for all COLD environments.")

if __name__ == "__main__":
    main()
