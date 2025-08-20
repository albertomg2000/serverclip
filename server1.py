import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms
import open_clip
from PIL import Image
import io
from typing import Optional

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo CLIP
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-H-14', pretrained='laion2b_s32b_b79k'
)
clip_model = clip_model.to(DEVICE)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

# Cargar embeddings de modelos (marca + modelo)
model_ckpt = torch.load("text_embeddings_modelos_h14.pt", map_location=DEVICE)
model_labels = model_ckpt["labels"]
model_embeddings = model_ckpt["embeddings"].to(DEVICE)
model_embeddings /= model_embeddings.norm(dim=-1, keepdim=True)

# Cargar embeddings de versiones (marca + modelo + versión)
version_ckpt = torch.load("text_embeddings_h14.pt", map_location=DEVICE)
version_labels = version_ckpt["labels"]
version_embeddings = version_ckpt["embeddings"].to(DEVICE)
version_embeddings /= version_embeddings.norm(dim=-1, keepdim=True)

# Transformación de imagen
normalize = next(t for t in preprocess.transforms if isinstance(t, transforms.Normalize))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=normalize.mean, std=normalize.std),
])

app = FastAPI()

def predict_top(text_feats, text_labels, image_tensor, topk=3):
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_feats.T).softmax(dim=-1)
        topk_result = torch.topk(similarity[0], k=topk)
    return [
        {
            "label": text_labels[idx],
            "confidence": round(conf.item() * 100, 2)
        }
        for conf, idx in zip(topk_result.values, topk_result.indices)
    ]

def process_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Paso 1: predecir modelo
    top_model = predict_top(model_embeddings, model_labels, img_tensor, topk=1)[0]
    modelo_predecido = top_model["label"]
    confianza_modelo = top_model["confidence"]

    # Separar marca y modelo
    marca, modelo = modelo_predecido.split(" ", 1)

    # Paso 2: buscar versiones que empiecen con ese modelo completo
    versiones_filtradas = [
        (label, idx) for idx, label in enumerate(version_labels)
        if label.startswith(modelo_predecido)
    ]

    if not versiones_filtradas:
        return {
            "marca": marca,
            "modelo": modelo,
            "confianza_modelo": confianza_modelo,
            "version": "No se encontraron versiones para este modelo"
        }

    # Extraer embeddings correspondientes
    indices_versiones = [idx for _, idx in versiones_filtradas]
    versiones_labels = [label for label, _ in versiones_filtradas]
    versiones_embeds = version_embeddings[indices_versiones]

    # Paso 3: predecir versión dentro de las versiones del modelo
    top_version = predict_top(versiones_embeds, versiones_labels, img_tensor, topk=1)[0]
    version_predicha = (
        top_version["label"].replace(modelo_predecido + " ", "") 
        if top_version["confidence"] >= 25
        else "Versión no identificada con suficiente confianza"
    )

    return {
        "marca": marca,
        "modelo": modelo,
        "confianza_modelo": confianza_modelo,
        "version": version_predicha,
        "confianza_version": top_version["confidence"]
    }


@app.post("/predict/")
async def predict(front: UploadFile = File(...), back: Optional[UploadFile] = File(None)):
    front_bytes = await front.read()
    if back:
        _ = await back.read()
    result = process_image(front_bytes)
    return JSONResponse(content=result)

