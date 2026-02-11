import numpy as np
from PIL import Image
import onnxruntime as ort
from scipy.special import expit

class ONNXModelAdapter:
    def __init__(self, onnx_path, mean, std, input_size):
        # Lista de provedores por ordem de prioridade
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Inicia a sessão ONNX
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        
        # Atributos de configuração do modelo
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.input_size = tuple(input_size) # (W, H)

    def predict(self, input_data):
        """Executa a inferência a partir do tensor já pré-processado."""
        logits = self.session.run(None, {self.input_name: input_data})[0]
        return expit(logits)

def preprocess_image(pil_image, target_size, mean, std):
    """
    Substitui as transformações do torchvision por operações NumPy.
    Agora recebe as constantes de normalização dinamicamente.
    """
    # 1. Redimensionamento via Pillow
    img_resized = pil_image.resize(target_size)
    
    # 2. Conversão para Float e Normalização
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_np = (img_np - np.array(mean)) / np.array(std)
    
    # 3. HWC -> CHW e Adição de dimensão de batch
    img_np = img_np.transpose(2, 0, 1)
    return np.expand_dims(img_np, axis=0)

def run_inference(adapter, pil_image, post_processor=None):
    """
    Pipeline unificado que utiliza as constantes do adaptador carregado.
    """
    # Pré-processamento usando os dados do adapter
    input_tensor = preprocess_image(
        pil_image, 
        adapter.input_size, 
        adapter.mean, 
        adapter.std
    )
    
    # Predição
    prob_map = adapter.predict(input_tensor)
    
    # Remove dimensões extras (Batch e Channel)
    prob_map_numpy = np.squeeze(prob_map)
    
    # Pós-processamento (Watershed, etc)
    if post_processor:
        return post_processor.process(prob_map_numpy)
    
    return prob_map_numpy