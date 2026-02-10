import cv2
import pandas as pd
import numpy as np
import os
from skimage.measure import label, regionprops
from typing import Optional

def calculate_area_scale_factor(
    original_width: int, 
    original_height: int, 
    mask_width: int = 1024, 
    mask_height: int = 768
) -> float:
    """
    Calculates the area scale factor between the original image and the model's mask.
    This factor represents how many original pixels fit into one mask pixel.
    """
    original_area = original_width * original_height
    mask_area = mask_width * mask_height
    
    if mask_area == 0:
        return 1.0
        
    return original_area / mask_area

def calculate_qwa_metrics(
    mask_path: str,
    original_w: int,
    original_h: int,
    pixel_size_um: float
) -> Optional[pd.DataFrame]:
    try:
        # 1. Leitura da máscara
        mascara = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mascara is None: 
            print(f"Erro ao ler máscara: {mask_path}")
            return None
            
        h_model, w_model = mascara.shape
        nome_imagem = os.path.basename(mask_path)

        # 2. Validação de Inputs
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("O parâmetro 'pixel_size_um' é obrigatório e deve ser > 0.")

        # 3. Cálculo dos Fatores de Escala usando a nova função
        fator_escala_area = calculate_area_scale_factor(original_w, original_h, w_model, h_model)
        fator_linear = np.sqrt(fator_escala_area) 
        
        area_dinamica_px_um2 = fator_escala_area * (pixel_size_um ** 2)
        tamanho_linear_px_um = fator_linear * pixel_size_um

        # --- Geometria ---
        pixel_area_mm2 = area_dinamica_px_um2 / 1_000_000.0
        lado_de_1mm_px = int(np.sqrt(1 / pixel_area_mm2)) if pixel_area_mm2 > 0 else 100
        centro_x, centro_y = w_model // 2, h_model // 2
        
        # ... (restante da lógica de quadrantes e extração de propriedades permanece igual)
        limites_quadrantes = {
            1: (centro_x - lado_de_1mm_px, centro_y - lado_de_1mm_px, centro_x, centro_y),
            2: (centro_x, centro_y - lado_de_1mm_px, centro_x + lado_de_1mm_px, centro_y),
            3: (centro_x - lado_de_1mm_px, centro_y, centro_x, centro_y + lado_de_1mm_px),
            4: (centro_x, centro_y, centro_x + lado_de_1mm_px, centro_y + lado_de_1mm_px)
        }

        def obter_quadrante(x, y):
            for num, (x0, y0, x1, y1) in limites_quadrantes.items():
                if x0 <= x < x1 and y0 <= y < y1: return num
            return 0

        # --- Extração ---
        mascara_rotulada = label(mascara, background=0)
        propriedades = regionprops(mascara_rotulada)

        lista_vasos = []
        for prop in propriedades:
            y_centroide, x_centroide = prop.centroid
            quadrante = obter_quadrante(x_centroide, y_centroide)
            y0_vaso, x0_vaso, y1_vaso, x1_vaso = prop.bbox

            is_inside_image = (x0_vaso > 0 and y0_vaso > 0 and 
                               x1_vaso < w_model and y1_vaso < h_model)

            # --- CÁLCULOS ---
            area_um2 = round((prop.area * area_dinamica_px_um2), 2)
            
            # Dados Brutos em Pixels
            major_axis_px = prop.axis_major_length
            minor_axis_px = prop.axis_minor_length
            
            # Conversão para µm
            major_axis_um = round(major_axis_px * tamanho_linear_px_um, 2)
            minor_axis_um = round(minor_axis_px * tamanho_linear_px_um, 2)

            lista_vasos.append({
                'Imagem': nome_imagem.split('_mask')[0],
                'Quadrante': quadrante,
                'Inside': is_inside_image,
                'Area_um2': area_um2,
                'Area_px': prop.area,
                'Major_Axis_um': major_axis_um,
                'Minor_Axis_um': minor_axis_um,
                'Major_Axis_px': major_axis_px, 
                'Minor_Axis_px': minor_axis_px, 
                'Centroide_X': x_centroide,
                'Centroide_Y': y_centroide
            })

        return pd.DataFrame(lista_vasos) if lista_vasos else pd.DataFrame()

    except Exception as e:
        print(f"Erro ao analisar máscara {mask_path}: {e}")
        return None