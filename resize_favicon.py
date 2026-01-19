from PIL import Image
import os

input_path = '/home/nick/projects/race_simulator/static/favicon.png'
output_png = '/home/nick/projects/race_simulator/static/favicon_small.png'
output_ico = '/home/nick/projects/race_simulator/static/favicon.ico'

if os.path.exists(input_path):
    img = Image.open(input_path)
    # Resize for PNG
    img_small = img.resize((32, 32), Image.Resampling.LANCZOS)
    img_small.save(output_png)
    
    # Save as ICO (can contain multiple sizes)
    img.save(output_ico, format='ICO', sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    print("Favicon resized and saved as PNG and ICO.")
else:
    print(f"Error: {input_path} not found.")
