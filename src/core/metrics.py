import cv2
import pandas as pd
import numpy as np
import os
from skimage.measure import label, regionprops
from typing import Optional

def calculate_area_scale_factor(
    image_shape: tuple, 
    mask_shape: tuple
) -> float:
    """
    Calcula o fator de escala de área entre a imagem original e a máscara.
    image_shape: (Width, Height) da imagem original.
    mask_shape: (Height, Width) obtido via mask.shape.
    """
    original_area = image_shape[0] * image_shape[1]
    mask_area = mask_shape[1] * mask_shape[0]
    
    if mask_area == 0:
        return 1.0
        
    return original_area / mask_area

def calculate_qwa_metrics(
    mask_path: str,
    image_shape: tuple,
    pixel_size_um: float
) -> Optional[pd.DataFrame]:
    """
    Realiza a extração de métricas QWA com correção de distorção de aspect ratio.
    Garante que a classificação de quadrantes alinhe com a visualização.
    """
    try:
        # 1. Leitura da máscara
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None: 
            print(f"Erro ao ler máscara: {mask_path}")
            return None
            
        h_model, w_model = mask.shape
        orig_w, orig_h = image_shape # (Largura, Altura) originais
        file_name = os.path.basename(mask_path)

        # 2. Validação de parâmetros
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("O parâmetro 'pixel_size_um' deve ser positivo.")
        
        # 3. Cálculo de Escalas Independentes (Correção Geométrica)
        # Determina quanto cada pixel da máscara representa da imagem original em cada eixo
        scale_w = w_model / orig_w
        scale_h = h_model / orig_h
        
        # Fator de área: um² por pixel da máscara
        fator_escala_area = (orig_w * orig_h) / (w_model * h_model)
        area_dinamica_px_um2 = fator_escala_area * (pixel_size_um ** 2)
        
        # Fator linear médio para eixos (Major/Minor Axis)
        tamanho_linear_px_um = np.sqrt(fator_escala_area) * pixel_size_um

        # 4. Definição do Grid de 1mm² (1000um x 1000um)
        # O lado de 1mm na escala da máscara é calculado para cada eixo separadamente
        lado_px_original = 1000 / pixel_size_um
        lado_px_w = int(lado_px_original * scale_w)
        lado_px_h = int(lado_px_original * scale_h)
        
        centro_x, centro_y = w_model // 2, h_model // 2
        
        # Coordenadas exatas dos quadrantes (idêntico ao visualization.py)
        limites = {
            1: (centro_x - lado_px_w, centro_y - lado_px_h, centro_x, centro_y),
            2: (centro_x, centro_y - lado_px_h, centro_x + lado_px_w, centro_y),
            3: (centro_x - lado_px_w, centro_y, centro_x, centro_y + lado_px_h),
            4: (centro_x, centro_y, centro_x + lado_px_w, centro_y + lado_px_h)
        }

        def obter_quadrante(x, y):
            for q_id, (x0, y0, x1, y1) in limites.items():
                if x0 <= x < x1 and y0 <= y < y1:
                    return q_id
            return 0 # Indica "Fora do Grid"

        # 5. Extração com Scikit-Image
        mascara_rotulada = label(mask, background=0)
        propriedades = regionprops(mascara_rotulada)

        lista_vasos = []
        for prop in propriedades:
            y_centroide, x_centroide = prop.centroid # Retorna (y, x)
            
            quadrante = obter_quadrante(x_centroide, y_centroide)
            y0, x0, y1, x1 = prop.bbox

            # Flag para vasos que tocam a borda da máscara
            is_inside_image = (x0 > 0 and y0 > 0 and x1 < w_model and y1 < h_model)

            # Conversão para unidades de Anatomia da Madeira (µm)
            area_um2 = round((prop.area * area_dinamica_px_um2), 2)
            major_axis_um = round(prop.axis_major_length * tamanho_linear_px_um, 2)
            minor_axis_um = round(prop.axis_minor_length * tamanho_linear_px_um, 2)

            lista_vasos.append({
                'Imagem': file_name.split('_mask')[0],
                'Quadrante': quadrante,
                'Inside': is_inside_image,
                'Area_um2': area_um2,
                'Area_px': prop.area,
                'Major_Axis_um': major_axis_um,
                'Minor_Axis_um': minor_axis_um,
                'Major_Axis_px': prop.axis_major_length, 
                'Minor_Axis_px': prop.axis_minor_length, 
                'Centroide_X': x_centroide,
                'Centroide_Y': y_centroide
            })

        return pd.DataFrame(lista_vasos)

    except Exception as e:
        print(f"Erro no processamento de métricas: {e}")
        return None