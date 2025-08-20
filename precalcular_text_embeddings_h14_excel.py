# precalcular_text_embeddings_h14_excel.py
import torch
import open_clip
import pandas as pd

# Leer el Excel
df = pd.read_excel("versiones_coche.xlsx")

# Crear los textos combinando marca, modelo y versión
def combinar_filas(row):
    if pd.isna(row["Version"]) or not row["Version"]:
        return f'{row["Marca"]} {row["Modelo"]}'
    return f'{row["Marca"]} {row["Modelo"]} {row["Version"]}'

textos = df.apply(combinar_filas, axis=1).tolist()

# Cargar modelo
model, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

# Calcular embeddings
with torch.no_grad():
    text_inputs = tokenizer(textos)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Guardar
torch.save({'embeddings': text_features, 'labels': textos}, 'text_embeddings_h14.pt')
print("Embeddings de texto guardados en 'text_embeddings_h14.pt'")
