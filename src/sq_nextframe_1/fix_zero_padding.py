import os

# Replace with your directory path
dir_path = "/home/vega/Coding/Training/ml-refresh/src/sq_nextframe_1/videos/secret_life_of_pets/28_28/frames"

for filename in os.listdir(dir_path):
    if filename.endswith(".png"):
        # Remove leading zeros
        new_filename = filename.lstrip("0")

        # Handle the case when filename is '00000.png'
        if new_filename[0] == ".":
            new_filename = "0" + new_filename

        os.rename(
            os.path.join(dir_path, filename), os.path.join(dir_path, new_filename)
        )
