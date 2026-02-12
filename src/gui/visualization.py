import cv2
import numpy as np
import pandas as pd
from typing import Optional

def draw_grid(
    mask_img: np.ndarray, 
    size_original: tuple,
    pixel_size_um: float, 
    df_vessels: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Redimensiona a máscara para o tamanho original e desenha o grid 1:1.
    """
    # 1. Redimensionamento direto para o tamanho original (sem bordas)
    # size_original deve ser (Width, Height)
    vis_img = cv2.resize(mask_img, size_original, interpolation=cv2.INTER_NEAREST)
    
    scale_w = size_original[0] / mask_img.shape[1]
    scale_h = size_original[1] / mask_img.shape[0]
    
    # Converte para BGR para desenhar colorido
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

    h_full, w_full = vis_img.shape[:2]
    cx, cy = w_full // 2, h_full // 2
    
    # 2. Cálculo do lado de 1mm na escala real
    # Como a imagem é a original, não há fator de escala de modelo aqui
    lado_px = int(1000 / pixel_size_um)
    
    # Cores dos Quadrantes
    cores = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 165, 0)}
    
    # Coordenadas do Grid
    quadrantes = {
        1: ((cx - lado_px, cy - lado_px), (cx, cy)),
        2: ((cx, cy - lado_px), (cx + lado_px, cy)),
        3: ((cx - lado_px, cy), (cx, cy + lado_px)),
        4: ((cx, cy), (cx + lado_px, cy + lado_px))
    }
    
    thickness = max(1, int(min(size_original) * 0.0025))
    circle = max(3, int(min(size_original) * 0.005))
    
    # 3. Desenhar Retângulos
    for q_id, (p1, p2) in quadrantes.items():
        cor = cores.get(q_id, (255, 255, 255))
        cv2.rectangle(vis_img, p1, p2, cor, thickness)

    # 4. Desenhar Centróides
    # IMPORTANTE: Se o df_vessels foi gerado na escala da máscara original (ex: 2584px),
    # as coordenadas x,y já estão prontas.
    if df_vessels is not None and not df_vessels.empty:
        # Se os centróides no CSV vieram da máscara pequena (ex: 640x640), 
        # você precisaria multiplicar x e y pelo fator de escala aqui.
        # Mas como combinamos de unificar para a escala original:
        for _, row in df_vessels.iterrows():
            try:
                x = int(row['Centroide_X'] * scale_w)
                y = int(row['Centroide_Y'] * scale_h)
                q = int(row['Quadrante'])
                cor_ponto = cores.get(q, (255, 0, 255)) # Magenta se erro de lógica
                cv2.circle(vis_img, (x, y), circle, cor_ponto, -1)
            except: continue

    return vis_img