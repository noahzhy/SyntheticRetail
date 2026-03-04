import json
import csv
import os
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def upload_images(img: str):
    """
    上传图片到文件服务器
    
    Args:
        img: 图片文件路径
        
    Returns:
        str: 上传后的图片URL
    """
    # send a request using PUT method with a given header
    filesvc_add_api = "https://fileman.clobotics.cn/api/add/base64"
    filesvs_file_api = "https://fileman.clobotics.cn/api/file/"

    headers = {
        "FileManAPIAccessToken": "Q2xvYm90aWNzLlJldGFpbC5CaXpNYW4uRmxvd01hbg==",
        "Content-Type": "application/json",
    }

    base64_str = base64.b64encode(open(img, "rb").read()).decode("utf-8")
    
    # 从路径中提取文件名
    img_name = Path(img).stem
    file_extension = Path(img).suffix if Path(img).suffix else ".jpg"

    data = {
        "AccessToken": "Q2xvYm90aWNzLlJldGFpbC5CaXpNYW4uRmxvd01hbg==",
        "FileContent": base64_str,
        "UploadFileInfo": {"Name": f"{img_name}{file_extension}"},
    }

    resp = requests.put(filesvc_add_api, headers=headers, json=data, timeout=30)
    resp.raise_for_status()
    
    decode_str = resp.content.decode("utf-8")
    res_json = json.loads(decode_str)
    
    if "FileId" not in res_json:
        raise Exception(f"上传失败，服务器响应: {res_json}")
    
    file_id = res_json["FileId"]
    file_url = filesvs_file_api + file_id
    return file_url


def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Convert a single JSON file to CSV format.
    
    Args:
        json_file_path: Path to the input JSON file
        csv_file_path: Path to the output CSV file
    """
    # Read JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Build a mapping from product SKU to box indices
    # PriceTag format: "PriceTag_SKU_{sku}_{number}"
    price_tag_map = {}  # Maps SKU to list of price tag box IDs
    
    for idx, item in enumerate(data):
        label = item['label']
        box_id = idx + 1
        
        # Check if this is a price tag
        if label.startswith('PriceTag_SKU_'):
            # Extract SKU from price tag label
            # Example: "PriceTag_SKU_s54483_013" -> "s54483"
            parts = label.split('_')
            if len(parts) >= 4:
                sku = parts[2]  # Get the SKU part (e.g., "s54483")
                if sku not in price_tag_map:
                    price_tag_map[sku] = []
                price_tag_map[sku].append(box_id)
    
    # Get the JSON file name for ImgUrl column
    img_url = Path(json_file_path).name
    # replace .json with .jpg
    img_url = img_url.replace('.json', '.jpg')
    
    Path(csv_file_path).parent.mkdir(parents=True, exist_ok=True)
    # Open CSV file for writing
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        # Write header
        writer.writerow(['ImgUrl', 'ProductId', 'xmin', 'ymin', 'xmax', 'ymax', 'BoxId', 'LinkedBoxIds'])
        
        # Write data rows
        for idx, item in enumerate(data):
            product_id = item['label']
            bbox = item['bbox']
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
            box_id = idx + 1
            
            # Find linked price tag box IDs
            linked_box_ids = ''
            
            # Extract SKU from product label (e.g., "s54453.001" -> "s54453")
            if '.' in product_id:
                sku = product_id.split('.')[0]
                if sku in price_tag_map:
                    # Join multiple price tag IDs with comma
                    linked_box_ids = ','.join(map(str, price_tag_map[sku]))
            
            writer.writerow([img_url, product_id, xmin, ymin, xmax, ymax, box_id, linked_box_ids])
    
    print(f"Converted {json_file_path} to {csv_file_path}")


def convert_directory_to_single_csv(input_dir, output_csv_file):
    """
    Convert all JSON files in a directory to a single CSV file.
    
    Args:
        input_dir: Directory containing JSON files
        output_csv_file: Path to the single output CSV file
    """
    # Find all JSON files
    json_files = sorted(Path(input_dir).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Create output directory if needed
    Path(output_csv_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Open CSV file for writing
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        # Write header once
        writer.writerow(['ImgUrl', 'ProductId', 'xmin', 'ymin', 'xmax', 'ymax', 'BoxId', 'LinkedBoxIds'])
        
        # Process each JSON file
        for json_file in json_files:
            print(f"Processing {json_file.name}...")
            
            # Read JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Build a mapping from product SKU to box indices
            price_tag_map = {}
            
            for idx, item in enumerate(data):
                label = item['label']
                box_id = idx + 1
                
                if label.startswith('PriceTag_SKU_'):
                    parts = label.split('_')
                    if len(parts) >= 4:
                        sku = parts[2]
                        if sku not in price_tag_map:
                            price_tag_map[sku] = []
                        price_tag_map[sku].append(box_id)
            
            # Get the image URL
            img_url = json_file.name.replace('.json', '.jpg')
            
            # Write data rows
            for idx, item in enumerate(data):
                product_id = item['label']
                bbox = item['bbox']
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                box_id = idx + 1
                
                # Find linked price tag box IDs
                linked_box_ids = ''
                
                if '.' in product_id:
                    sku = product_id.split('.')[0]
                    if sku in price_tag_map:
                        linked_box_ids = ','.join(map(str, price_tag_map[sku]))
                
                writer.writerow([img_url, product_id, xmin, ymin, xmax, ymax, box_id, linked_box_ids])
    
    print(f"\nAll files converted to {output_csv_file}")


def convert_directory(input_dir, output_dir):
    """
    Convert all JSON files in a directory to CSV format.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    json_files = list(Path(input_dir).glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Convert each JSON file
    for json_file in json_files:
        csv_file = Path(output_dir) / f"{json_file.stem}.csv"
        convert_json_to_csv(json_file, csv_file)
    
    print(f"\nAll files converted successfully!")


def convert_all_sequences_to_single_csv(labels_dir, output_csv_file):
    """
    Convert all JSON files in all sequence directories to a single CSV file.
    
    Args:
        labels_dir: Base labels directory containing sequence folders
        output_csv_file: Path to the single output CSV file
    """
    labels_path = Path(labels_dir)
    
    # Find all sequence directories
    sequence_dirs = sorted([d for d in labels_path.iterdir() if d.is_dir() and d.name.startswith('sequence_')])
    
    if not sequence_dirs:
        print(f"No sequence directories found in {labels_dir}")
        return
    
    print(f"Found {len(sequence_dirs)} sequence directories")
    
    # Create output directory if needed
    Path(output_csv_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Open CSV file for writing
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        
        # Write header once
        writer.writerow(['ImgUrl', 'ProductId', 'xmin', 'ymin', 'xmax', 'ymax', 'BoxId', 'LinkedBoxIds'])
        
        # Process each sequence directory
        for seq_dir in sequence_dirs:
            print(f"\nProcessing {seq_dir.name}...")
            
            # Find all JSON files in this sequence
            json_files = sorted(seq_dir.glob('*.json'))
            print(f"  Found {len(json_files)} JSON files")
            
            # Process each JSON file
            for json_file in json_files:
                print(f"  Processing {json_file.name}...")
                
                # Read JSON file
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Build a mapping from product SKU to box indices
                price_tag_map = {}
                
                for idx, item in enumerate(data):
                    label = item['label']
                    box_id = idx + 1
                    
                    if label.startswith('PriceTag_SKU_'):
                        parts = label.split('_')
                        if len(parts) >= 4:
                            sku = parts[2]
                            if sku not in price_tag_map:
                                price_tag_map[sku] = []
                            price_tag_map[sku].append(box_id)
                
                # Get the image URL
                img_url = json_file.name.replace('.json', '.jpg')
                
                # Write data rows
                for idx, item in enumerate(data):
                    product_id = item['label']
                    bbox = item['bbox']
                    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                    box_id = idx + 1
                    
                    # Find linked price tag box IDs
                    linked_box_ids = ''
                    
                    if '.' in product_id:
                        sku = product_id.split('.')[0]
                        if sku in price_tag_map:
                            linked_box_ids = ','.join(map(str, price_tag_map[sku]))
                    
                    writer.writerow([img_url, product_id, xmin, ymin, xmax, ymax, box_id, linked_box_ids])
    
    print(f"\nAll sequences converted to {output_csv_file}")


def convert_all_sequences(labels_dir, output_base_dir):
    """
    Convert all JSON files in all sequence directories.
    
    Args:
        labels_dir: Base labels directory containing sequence folders
        output_base_dir: Base output directory for CSV files
    """
    labels_path = Path(labels_dir)
    
    # Find all sequence directories
    sequence_dirs = [d for d in labels_path.iterdir() if d.is_dir() and d.name.startswith('sequence_')]
    
    if not sequence_dirs:
        print(f"No sequence directories found in {labels_dir}")
        return
    
    print(f"Found {len(sequence_dirs)} sequence directories")
    
    # Process each sequence directory
    for seq_dir in sequence_dirs:
        print(f"\nProcessing {seq_dir.name}...")
        output_dir = Path(output_base_dir) / seq_dir.name
        convert_directory(seq_dir, output_dir)


def merge_csv_files(input_dir, output_file):
    """
    Merge all CSV files in a directory and its subdirectories into one file.
    
    Args:
        input_dir: Directory containing CSV files
        output_file: Path to the merged output CSV file
    """
    input_path = Path(input_dir)
    
    # Find all CSV files recursively
    csv_files = list(input_path.rglob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to merge")
    
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Open output file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = None
        header_written = False
        
        for csv_file in sorted(csv_files):
            print(f"Merging {csv_file.name}...")
            
            with open(csv_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                
                for i, row in enumerate(reader):
                    if i == 0:  # Header row
                        if not header_written:
                            writer = csv.writer(outfile, delimiter=',')
                            writer.writerow(row)
                            header_written = True
                    else:  # Data rows
                        writer.writerow(row)
    
    print(f"\nMerged {len(csv_files)} files into {output_file}")


if __name__ == "__main__":
    # Example usage - convert a single file
    # convert_json_to_csv(
    #     'labels/sequence_20260206_152444/frame_0001_20260206_152444.json',
    #     'output/frame_0001_20260206_152444.csv'
    # )
    
    # # Convert all files in a sequence directory to a single CSV
    # convert_directory_to_single_csv(
    #     'labels/sequence_20260206_152444',
    #     'output/sequence_20260206_152444.csv'
    # )
    
    # Convert all sequences to a single CSV file
    convert_all_sequences_to_single_csv(
        'labels',
        'output/all_labels.csv'
    )

    # find local images and upload to file server and replace ImgUrl in the CSV
    # avoid duplicate upload same image
    csv_file = 'output/all_labels.csv'
    temp_csv_file = 'output/all_labels_temp.csv'
    
    # First, read all unique image URLs
    unique_images = set()
    with open(csv_file, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        for row in reader:
            unique_images.add(row[0])
    
    print(f"Found {len(unique_images)} unique images to upload")
    
    # Upload images in parallel
    img_url_map = {}  # local image path to uploaded URL
    
    def upload_single_image(img_filename):
        try:
            img_path = os.path.join('images', img_filename)
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                return img_filename, None
            uploaded_url = upload_images(img_path)
            print(f"Uploaded {img_filename} -> {uploaded_url}")
            return img_filename, uploaded_url
        except Exception as e:
            print(f"Error uploading {img_filename}: {e}")
            return img_filename, None
    
    # Use ThreadPoolExecutor for parallel uploads
    max_workers = 10  # Adjust based on server capacity
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(upload_single_image, img): img for img in unique_images}
        
        for future in as_completed(future_to_img):
            img_filename, uploaded_url = future.result()
            if uploaded_url:
                img_url_map[img_filename] = uploaded_url
    
    print(f"\nSuccessfully uploaded {len(img_url_map)} images")
    
    # Now write the CSV with uploaded URLs
    with open(csv_file, 'r', encoding='utf-8') as infile, open(temp_csv_file, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, delimiter=',')
        
        for i, row in enumerate(reader):
            if i == 0:
                # Header row
                writer.writerow(row)
            else:
                img_url = row[0]
                if img_url in img_url_map:
                    row[0] = img_url_map[img_url]
                else:
                    print(f"Warning: No uploaded URL for {img_url}")
                writer.writerow(row)
    
    # Replace original file with temp file
    os.replace(temp_csv_file, csv_file)
    print(f"\nUpdated {csv_file} with uploaded image URLs")
    

