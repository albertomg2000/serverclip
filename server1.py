import os
import io
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image

# Lazy imports (se importan al cargar el modelo)
open_clip = None

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Estado global (caché en memoria)
_state = {
    "clip_model": None,
    "transform": None,
    "model_embeddings": None,
    "model_labels": None,
    "version_embeddings": None,
    "version_labels": None,
}

def _load_runtime():
    """Carga el modelo y los embeddings solo una vez (primer request)."""
    global open_clip
    if _state["clip_model"] is not None:
        return

    # Import aquí para que no gaste RAM hasta que haga falta
    import open_clip as _open_clip
    from torchvision import transforms as T
    open_clip = _open_clip

    # Modelo pequeño para Render / RAM baja
    vit_name = os.environ.get("CLIP_VIT", "ViT-B-32")
    pretrained = os.environ.get("CLIP_PRETRAINED", "openai")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        vit_name, pretrained=pretrained
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Embeddings
    model_ckpt = torch.load("text_embeddings_modelos_h14.pt", map_location=DEVICE)
    version_ckpt = torch.load("text_embeddings_h14.pt", map_location=DEVICE)

    model_emb = model_ckpt["embeddings"].to(DEVICE)
    model_emb = model_emb / model_emb.norm(dim=-1, keepdim=True)
    version_emb = version_ckpt["embeddings"].to(DEVICE)
    version_emb = version_emb / version_emb.norm(dim=-1, keepdim=True)

    # Transform
    normalize = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=normalize.mean, std=normalize.std),
    ])

    _state.update({
        "clip_model": clip_model,
        "transform": transform,
        "model_embeddings": model_emb,
        "model_labels": model_ckpt["labels"],
        "version_embeddings": version_emb,
        "version_labels": version_ckpt["labels"],
    })

def _predict_top(text_feats, text_labels, image_tensor, topk=3):
    with torch.no_grad():
        image_features = _state["clip_model"].encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_feats.T).softmax(dim=-1)
        topk_result = torch.topk(similarity[0], k=topk)
    return [
        {"label": text_labels[idx], "confidence": round(conf.item() * 100, 2)}
        for conf, idx in zip(topk_result.values, topk_result.indices)
    ]

def _process_image(image_bytes: bytes):
    _load_runtime()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = _state["transform"](img).unsqueeze(0).to(DEVICE)

    # Paso 1: modelo
    top_model = _predict_top(_state["model_embeddings"], _state["model_labels"], img_tensor, topk=1)[0]
    modelo_predecido = top_model["label"]
    confianza_modelo = top_model["confidence"]

    # Marca + modelo
    try:
        marca, modelo = modelo_predecido.split(" ", 1)
    except ValueError:
        marca, modelo = "", modelo_predecido

    # Filtrar versiones que empiecen por el label completo del modelo
    versiones_filtradas = [
        (label, idx) for idx, label in enumerate(_state["version_labels"])
        if label.startswith(modelo_predecido)
    ]
    if not versiones_filtradas:
        return {
            "marca": marca,
            "modelo": modelo,
            "confianza_modelo": confianza_modelo,
            "version": "No se encontraron versiones para este modelo"
        }

    indices_versiones = [idx for _, idx in versiones_filtradas]
    versiones_labels = [label for label, _ in versiones_filtradas]
    versiones_embeds = _state["version_embeddings"][indices_versiones]

    # Paso 3: versión
    top_version = _predict_top(versiones_embeds, versiones_labels, img_tensor, topk=1)[0]
    version_predicha = (
        top_version["label"].replace(modelo_predecido + " ", "")
        if top_version["confidence"] >= 25 else
        "Versión no identificada con suficiente confianza"
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
        _ = await back.read()  # (reservado para futuro)
    result = _process_image(front_bytes)
    return JSONResponse(content=result)

