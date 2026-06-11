from zipfile import ZipFile

# Path to the ZIP file
file_name = "best.zip"

# Open in read mode and extract all contents
with ZipFile(file_name, 'r') as zip_ref:
	zip_ref.printdir() # List contents
	zip_ref.extractall() # Extract to current directory

print("Extraction complete!")
