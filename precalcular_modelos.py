import torch
import open_clip
import pandas as pd

# Solo con marca + modelo
df = pd.read_excel("modelos.xlsx")

textos = (df["Marca"] + " " + df["Modelo"]).tolist()

# Cargar modelo y tokenizer
model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

# Generar embeddings
with torch.no_grad():
    text_inputs = tokenizer(textos)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Guardar
torch.save({'embeddings': text_features, 'labels': textos}, 'text_embeddings_modelos_h14.pt')
print("Embeddings de modelos guardados en 'text_embeddings_modelos_h14.pt'")
