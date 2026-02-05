import cv2
import numpy as np
import pandas as pd
from typing import Optional

def desenhar_grid_quadrantes(
    mask_img: np.ndarray, 
    pixel_size_um: float, 
    original_w: int,
    df_vessels: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Desenha os 4 quadrantes centrais de 1mm² na imagem da máscara,
    ajustando a escala dos pixels da imagem original para a máscara.
    
    Args:
        mask_img: Máscara binária ou grayscale (ex: 1024x768).
        pixel_size_um: Resolução da imagem ORIGINAL (µm/pixel).
        original_w: Largura da imagem ORIGINAL (ex: 2584).
        df_vessels: (Opcional) DataFrame com centróides (já nas coordenadas da máscara).
    """
    # Converte para BGR
    if len(mask_img.shape) == 2:
        vis_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = mask_img.copy()

    # Dimensões da Máscara (Pequena, ex: 1024x768)
    h_mask, w_mask = vis_img.shape[:2]
    cx, cy = w_mask // 2, h_mask // 2
    
    # --- CORREÇÃO DA ESCALA ---
    # 1. Quantos pixels tem 1mm na imagem ORIGINAL?
    lado_px_original = 1000 / pixel_size_um  # ~940 px
    
    # 2. Qual foi o fator de redução da largura (Original -> Máscara)?
    # Ex: 1024 / 2584 = 0.39
    scale_factor = w_mask / original_w
    
    # 3. Quantos pixels tem 1mm na MÁSCARA?
    # Ex: 940 * 0.39 = ~366 px
    lado_px = int(lado_px_original * scale_factor)
    
    # --------------------------
    
    # Cores (BGR) - Q1: Azul, Q2: Verde, Q3: Vermelho, Q4: Laranja
    cores_quadrantes = {
        1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 165, 0)
    }
    
    # Coordenadas Geométricas (Ancoradas no Centro)
    quadrantes_coords = {
        1: ((cx - lado_px, cy - lado_px), (cx, cy)),
        2: ((cx, cy - lado_px), (cx + lado_px, cy)),
        3: ((cx - lado_px, cy), (cx, cy + lado_px)),
        4: ((cx, cy), (cx + lado_px, cy + lado_px))
    }
    
    # Configuração Visual
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8 # Reduzi um pouco pois a máscara é menor

    # 1. Desenhar Retângulos e Rótulos
    for q_id, (p1, p2) in quadrantes_coords.items():
        cor = cores_quadrantes.get(q_id, (255, 255, 255))
        cv2.rectangle(vis_img, p1, p2, cor, thickness)
        

    # 3. Centróides dos Vasos
    # Nota: As coordenadas no df_vessels já vêm do metrics.py baseadas na máscara (regionprops na mask),
    # então NÃO precisamos escalar as posições X,Y, elas já estão certas para 1024x768.
    if df_vessels is not None and not df_vessels.empty:
        for _, row in df_vessels.iterrows():
            try:
                x = int(row['Centroide_X'])
                y = int(row['Centroide_Y'])
                q = row['Quadrante']
                cor_ponto = cores_quadrantes.get(q, (255, 0, 255))
                cv2.circle(vis_img, (x, y), 5, cor_ponto, -1)
            except Exception:
                continue

    return vis_img