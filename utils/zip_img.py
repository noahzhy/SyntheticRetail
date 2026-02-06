import os
import zipfile
import sys

def zip_images(folder_path, zip_name):
    # List of common image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)
                    # Add file to zip without subfolder structure
                    zipf.write(file_path, file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python zip_img.py <folder_path> <zip_name>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    zip_name = sys.argv[2]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        sys.exit(1)
    
    zip_images(folder_path, zip_name)
    print(f"Images from {folder_path} have been zipped into {zip_name}")