import torch
import torchvision.transforms as transforms
from PIL import Image

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
materials = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
recycle = ["cardboard", "paper", "metal"]
waste = ["glass", "plastic", "trash"]
processor = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def predict_image(image_path, model):
    img = Image.open(image_path)
    transformed_image = transformations(img)
    
    results = model(to_device(transformed_image.unsqueeze(0), processor))
    _, preds = torch.max(results, dim=1)
    return materials[preds[0].item()]

def is_recyclable(material):
    if material in recycle:
        return True
    else:
        return False
