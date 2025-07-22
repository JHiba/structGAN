import os
import scipy.io
import numpy as np
import cv2
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
import torchvision.utils as vutils
from pix import SimpleUNet  # Your model class

##### --------- PART 1: .mat to floorplan PNG --------- #####
def create_door_line(x, y, width, height, orientation):
    # Returns a line segment for a door/window
    if width > 0 and height == 0:
        if orientation == 0: return [(int(x), int(y)), (int(x + width), int(y))]
        elif orientation == 2: return [(int(x), int(y)), (int(x - width), int(y))]
        else: return [(int(x), int(y)), (int(x + width), int(y))]
    elif height > 0 and width == 0:
        if orientation == 1: return [(int(x), int(y)), (int(x), int(y + height))]
        elif orientation == 3: return [(int(x), int(y)), (int(x), int(y - height))]
        else: return [(int(x), int(y)), (int(x), int(y + height))]
    else:
        return [(int(x), int(y)), (int(x), int(y))]

def mat_to_structgan_png(mat_path, output_path="structgan_input.png", img_size=256):
    mat = scipy.io.loadmat(mat_path)
    data = mat['data'][0, 0]
    polygons = data['rBoundary'][0]

    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background

    # Draw walls as gray lines
    for poly in polygons:
        poly = np.array(poly)
        if poly.size == 0 or poly.shape[0] < 2:
            continue
        pts = poly.astype(np.int32)
        for j in range(len(pts)):
            pt1 = tuple(pts[j % len(pts)])
            pt2 = tuple(pts[(j + 1) % len(pts)])
            cv2.line(canvas, pt1, pt2, (132, 132, 132), 3)  # Gray wall

    # Draw doors as blue lines
    if 'doors' in data.dtype.fields:
        for door_entry in np.array(data['doors']).reshape(-1, 6):
            _, x, y, width, height, orientation = door_entry.astype(float)
            pt1, pt2 = create_door_line(x, y, width, height, int(orientation))
            cv2.line(canvas, pt1, pt2, (255, 0, 0), 4)  # Blue

    # Draw windows as green lines
    if 'windows' in data.dtype.fields:
        for win_entry in np.array(data['windows']).reshape(-1, 6):
            _, x, y, width, height, orientation = win_entry.astype(float)
            pt1, pt2 = create_door_line(x, y, width, height, int(orientation))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 4)  # Green

    cv2.imwrite(output_path, canvas)
    print(f"Saved: {output_path}")

def batch_convert_mat_to_png(mat_dir, output_dir="structgan_pngs", img_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mat_files = glob(os.path.join(mat_dir, "*.mat"))
    print(f"Found {len(mat_files)} .mat files.")

    for mat_path in mat_files:
        base = os.path.splitext(os.path.basename(mat_path))[0]
        png_path = os.path.join(output_dir, f"{base}.png")
        try:
            mat_to_structgan_png(mat_path, png_path, img_size=img_size)
        except Exception as e:
            print(f"Error with {mat_path}: {e}")

##### --------- PART 2: Run Model on Floorplan PNGs --------- #####
def run_model_on_pngs(input_dir, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    model = SimpleUNet(input_nc=3, output_nc=3, ngf=64).to(device)
    model.load_state_dict(torch.load('outputs_pix2pix1/generator.pth', map_location=device))
    model.eval()

    def denorm(tensor):
        return tensor * 0.5 + 0.5

    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, fname)
            image = Image.open(input_path).convert('RGB')
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output_tensor = model(img_tensor)
                output_tensor = denorm(output_tensor.squeeze(0)).clamp(0, 1).cpu()

            save_path = os.path.join(output_dir, fname)
            vutils.save_image(output_tensor, save_path)
            print(f"Processed {fname} -> saved output to {save_path}")

    print("All images processed!")

##### --------- MAIN --------- #####
if __name__ == "__main__":
    mat_dir = r"C:/Users/graan/structgan/matfiles"
    png_output_dir = "floorplan_pngs"
    custom_output_dir = 'outputs'
    img_size = 256

    # 1. Convert .mat files to floorplan pngs
    batch_convert_mat_to_png(mat_dir, png_output_dir, img_size=img_size)

    # 2. Run model on all those pngs
    run_model_on_pngs(png_output_dir, custom_output_dir)
