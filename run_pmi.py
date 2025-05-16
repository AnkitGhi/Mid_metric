import csv

def create_csv_from_txt(txt_file_path, csv_file_path):
    """
    Reads a text file with image names and captions and creates a CSV file.

    Args:
        txt_file_path (str): The path to the input text file.
        csv_file_path (str): The path to the output CSV file.
    """
    with open(txt_file_path, 'r') as infile, open(csv_file_path, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        # Write the header row for the CSV file
        csv_writer.writerow(['image', 'caption'])

        for line in infile:
            # Strip any leading/trailing whitespace
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Split the line at the first tab character
            parts = line.split('\t', 1)
            if len(parts) == 2:
                image_info, caption = parts
                # Extract the part before '#'
                image_name = image_info.split('#')[0]
                # Construct the full image path
                image_path = f"drive/MyDrive/Flickr8k/{image_name}"
                csv_writer.writerow([image_path, caption])
            else:
                print(f"Skipping malformed line: {line}")

# --- Example Usage ---
# Replace 'input.txt' with the actual path to your text file
# Replace 'output.csv' with the desired path for your CSV file
input_file = 'annotations.txt'
output_file = 'output.csv'


create_csv_from_txt(input_file, output_file)

print(f"CSV file '{output_file}' created successfully.")

# You can also print the content of the output file to verify:
# with open(output_file, 'r') as f:
#     print("\nContent of the output CSV file:")
#     print(f.read())