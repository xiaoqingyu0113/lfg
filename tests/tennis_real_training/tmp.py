import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import glob

# Directory containing the images
image_dir = "plots/increm_gtsam"  # Replace with the directory where your PNG files are stored
output_gif = "increm_gtsam.gif"  # Name of the output GIF
frame_duration = 100  # Duration of each frame in milliseconds

# List to store paths to images
image_files = glob.glob(f"{image_dir}/*.png")
image_files.sort()  # Sort the files to maintain order

# Create a list to store the frames for the GIF
frames = []

for img_file in image_files[::6]:
    # Load the saved image into Pillow
    frame = Image.open(img_file)
    frames.append(frame)

# Save the frames as a GIF
frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    duration=frame_duration,
    loop=0  # Infinite loop
)