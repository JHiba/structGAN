import io
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import scipy.io
import cv2
from PIL import Image
from torchvision import transforms
from pix import SimpleUNet
import os
app = FastAPI()

# Model + transform setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleUNet(input_nc=3, output_nc=3, ngf=64).to(device)
model.load_state_dict(torch.load('outputs_pix2pix1/generator.pth', map_location=device))
model.eval()
img_size = 256
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
def denorm(tensor):
    return tensor * 0.5 + 0.5

def create_door_line(x, y, width, height, orientation):
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

def mat_bytes_to_png_bytes(mat_bytes, img_size=256):
    mat = scipy.io.loadmat(io.BytesIO(mat_bytes))
    data = mat['data'][0, 0]
    polygons = data['rBoundary'][0]

    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    for poly in polygons:
        poly = np.array(poly)
        if poly.size == 0 or poly.shape[0] < 2:
            continue
        pts = poly.astype(np.int32)
        for j in range(len(pts)):
            pt1 = tuple(pts[j % len(pts)])
            pt2 = tuple(pts[(j + 1) % len(pts)])
            cv2.line(canvas, pt1, pt2, (132, 132, 132), 3)

    if 'doors' in data.dtype.fields:
        for door_entry in np.array(data['doors']).reshape(-1, 6):
            _, x, y, width, height, orientation = door_entry.astype(float)
            pt1, pt2 = create_door_line(x, y, width, height, int(orientation))
            cv2.line(canvas, pt1, pt2, (255, 0, 0), 4)

    if 'windows' in data.dtype.fields:
        for win_entry in np.array(data['windows']).reshape(-1, 6):
            _, x, y, width, height, orientation = win_entry.astype(float)
            pt1, pt2 = create_door_line(x, y, width, height, int(orientation))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 4)

    # Encode as PNG bytes
    success, buf = cv2.imencode('.png', canvas)
    if not success:
        raise ValueError("Could not encode PNG.")
    return buf.tobytes()

def model_infer_png_bytes(png_bytes):
    image = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = model(img_tensor)
        output_tensor = denorm(output_tensor.squeeze(0)).clamp(0, 1).cpu()
    out_img = transforms.ToPILImage()(output_tensor)
    buf = io.BytesIO()
    out_img.save(buf, format='PNG')
    buf.seek(0)
    return buf

@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    mat_bytes = await file.read()
    try:
        floorplan_png_bytes = mat_bytes_to_png_bytes(mat_bytes, img_size=img_size)
    except Exception as e:
        return JSONResponse({"error": f"Failed to process {file.filename}: {e}"}, status_code=400)
    output_buf = model_infer_png_bytes(floorplan_png_bytes)
    # Filename for output
    base = os.path.splitext(file.filename)[0]
    output_name = f"{base}_output.png"
    return StreamingResponse(output_buf, media_type="image/png", headers={"Content-Disposition": f"attachment; filename={output_name}"})
