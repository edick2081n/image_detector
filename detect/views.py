from django.shortcuts import render
from torchvision.io import read_image
from torchvision.models import ResNet50_Weights, resnet50
from PIL import Image
from .forms import ImageUploadForm



weights = ResNet50_Weights.DEFAULT
model=resnet50(weights=weights)
model.eval()
preprocess = weights.transforms(antialias=True)

def index(request):
    form = ImageUploadForm()
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = Image.open(request.FILES["image"])
            batch = preprocess(image).unsqueeze(0)
            prediction = model(batch).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            category_name = weights.meta["categories"][class_id]
            width, height = image.size
    else:
        category_name=""
        width=""
        height=""
    return render(request, "index.html", {"form":form, "category_name": category_name, "width":width, "height":height})






