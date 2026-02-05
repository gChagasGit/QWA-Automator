import torch
from torchvision import transforms
import numpy as np

# Tenta importar o PostProcessor
try:
    from src.core.post_processing import MaskPostProcessor
except ImportError:
    pass

NORM_MEAN = [0.7599, 0.2988, 0.2935]
NORM_STD = [0.1335, 0.2212, 0.1162]
INPUT_SIZE = (768, 1024)

class LocalModelAdapter:
    """
    Classe que envolve o modelo PyTorch puro para ser usado na inferência.
    Trata Deep Supervision e Sigmoid automaticamente.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, input_tensor):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # 1. Trata Deep Supervision (Se o modelo retornar lista [out, d1, d2...])
            if isinstance(output, list):
                output = output[0]
            
            # 2. Aplica Sigmoid (Logits -> Probabilidade 0.0 a 1.0)
            return torch.sigmoid(output)

# --- FUNÇÕES DE API ---

def preprocess_image(pil_image, target_size=INPUT_SIZE):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    return transform(pil_image).unsqueeze(0)

def run_inference(model_ensemble, pil_image, post_processor=None, threshold=0.5):
    """
    Pipeline Completo atualizado para usar o novo PostProcessor.
    """
    # 1. Setup
    w, h = pil_image.size
    device = model_ensemble.device
    img_tensor = preprocess_image(pil_image).to(device)
    
    # 2. Predição (Ensemble)
    prob_map = model_ensemble.predict(img_tensor)
    
    # Converte para Numpy (Float 0.0 - 1.0)
    # Shape final: (H, W)
    prob_map_numpy = prob_map.squeeze().cpu().numpy()
    
    # 4. Pós-Processamento e Binarização
    # O seu novo PostProcessor lida com o threshold e watershed internamente
    if post_processor:
        # Se o processador já tiver threshold interno configurado, usa ele
        # Caso contrário, o código abaixo garante que ele receba o mapa de probabilidade
        mask_final = post_processor.process(prob_map_numpy)
    else:
        # Fallback simples se não houver processador
        mask_final = (prob_map_numpy > threshold).astype(np.uint8) * 255
        
    return mask_final