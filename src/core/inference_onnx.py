import numpy as np
from PIL import Image
import onnxruntime as ort
from scipy.special import expit

# Constantes de normalização conforme o treinamento original
NORM_MEAN = np.array([0.7599, 0.2988, 0.2935], dtype=np.float32)
NORM_STD = np.array([0.1335, 0.2212, 0.1162], dtype=np.float32)
INPUT_SIZE = (1024, 768) # Formato (Largura, Altura) para redimensionamento PIL

class ONNXModelAdapter:
    def __init__(self, model_path):
        # Lista de provedores por ordem de prioridade
        providers = [ 
            'CUDAExecutionProvider', 
            'CPUExecutionProvider'
        ]
        
        # Inicia a sessão. O ONNX fará o fallback sozinho se o TensorRT falhar
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Captura qual provedor foi realmente selecionado para uso
        # Adicionamos o atributo 'provider' para compatibilidade com seu script
        self.provider = self.session.get_providers()[0]
        
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_data):
        # Utiliza o expit para evitar os avisos de overflow que vimos antes
        logits = self.session.run(None, {self.input_name: input_data})[0]
        return expit(logits)

def preprocess_image(pil_image, target_size=INPUT_SIZE):
    """
    Substitui as transformações do torchvision por operações NumPy.
    """
    # 1. Redimensionamento via Pillow
    img_resized = pil_image.resize(target_size)
    
    # 2. Conversão para Float e Normalização
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_np = (img_np - NORM_MEAN) / NORM_STD
    
    # 3. HWC (Height, Width, Channel) -> CHW (Channel, Height, Width)
    img_np = img_np.transpose(2, 0, 1)
    
    # 4. Adiciona dimensão de batch (1, C, H, W)
    return np.expand_dims(img_np, axis=0)

def run_inference(adapter, pil_image, post_processor=None, threshold=0.5):
    """
    Pipeline unificado de inferência para segmentação de vasos.
    """
    # Pré-processamento
    input_tensor = preprocess_image(pil_image)
    
    # Predição direta via ONNX
    prob_map = adapter.predict(input_tensor)
    
    # Squeeze para remover dimensões de batch/canal (1, 1, H, W) -> (H, W)
    prob_map_2d = np.squeeze(prob_map)
    
    # Pós-processamento (Morfologia e Watershed)
    if post_processor:
        mask_final = post_processor.process(prob_map_2d)
    else:
        # Fallback de binarização simples
        mask_final = (prob_map_2d > threshold).astype(np.uint8) * 255
        
    return mask_final