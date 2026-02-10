import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops # Novos imports necessários

class MaskPostProcessor:

    KERNEL_3x3 = np.ones((3, 3), np.uint8)

    def __init__(self, threshold=0.5, min_area=100):
        self.threshold = threshold
        self.min_area = min_area

    @staticmethod
    def _is_touching_border(contour, shape):
        h, w = shape[:2]
        x, y, cw, ch = cv2.boundingRect(contour)
        return x <= 1 or y <= 1 or (x + cw) >= w - 1 or (y + ch) >= h - 1

    @staticmethod
    def _preprocess_mask(mask_prob_numpy, threshold):
        """
        Binariza e preenche buracos. 
        Removi a erosão inicial para não encolher os vasos antes da medição de área.
        """
        mask_bin = (mask_prob_numpy > threshold).astype(np.uint8) * 255
        
        # Preenchimento de buracos é vital para QWA
        mask_filled_bool = ndimage.binary_fill_holes(mask_bin > 0)
        return (mask_filled_bool.astype(np.uint8) * 255)

    @staticmethod
    def _calculate_split_threshold(valid_props, shape):
        """
        Calcula o limiar usando regionprops.area.
        """
        areas_validas = []
        for prop in valid_props:
            # Recriamos um contorno temporário apenas para checar a borda com sua função original
            component_mask = (prop.image).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts:
                # Ajustamos a coordenada do contorno para a posição global na imagem
                r0, c0, r1, c1 = prop.bbox
                cnt_global = cnts[0] + [c0, r0]
                if not MaskPostProcessor._is_touching_border(cnt_global, shape):
                    areas_validas.append(prop.area)

        if not areas_validas:
            return float('inf')

        return np.mean(areas_validas) + np.std(areas_validas)

    @staticmethod
    def _apply_watershed_split(contour, mask_shape):
        mask_roi = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(mask_roi, [contour], -1, 255, -1)

        dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)
        coords = peak_local_max(dist, min_distance=15, labels=mask_roi)

        if len(coords) < 2:
            return mask_roi

        markers = np.zeros(mask_shape, dtype=np.int32)
        for i, (r, c) in enumerate(coords):
            markers[r, c] = i + 2

        sure_bg = cv2.dilate(mask_roi, MaskPostProcessor.KERNEL_3x3, iterations=3)
        markers[sure_bg == 0] = 1 

        mask_roi_bgr = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_roi_bgr, markers)

        mask_separated = np.zeros(mask_shape, dtype=np.uint8)
        mask_separated[markers > 1] = 255
        cuts = (markers == -1).astype(np.uint8) * 255
        cuts_dilated = cv2.dilate(cuts, MaskPostProcessor.KERNEL_3x3, iterations=1)
        mask_separated[cuts_dilated > 0] = 0

        return mask_separated

    @staticmethod
    def refine(mask_prob_numpy, shape, threshold=0.5, min_area=100):
        # 1. Preprocessamento (Binarização + Fill Holes)
        mask_filled = MaskPostProcessor._preprocess_mask(mask_prob_numpy, threshold)
        
        # 2. Rotulagem para usar regionprops (Igual ao metrics.py)
        labeled_mask = label(mask_filled, background=0)
        props = regionprops(labeled_mask)
        
        # 3. FILTRAGEM POR ÁREA REAL DE PIXELS
        valid_props = [p for p in props if p.area >= min_area]
        
        # 4. Cálculo do limiar dinâmico para Watershed
        threshold_split = MaskPostProcessor._calculate_split_threshold(valid_props, shape)

        mask_final = np.zeros_like(mask_filled)

        for p in valid_props:
            # Isola o componente atual
            component_mask = (labeled_mask == p.label).astype(np.uint8) * 255
            
            if p.area < threshold_split:
                # Objeto de tamanho normal
                mask_final = cv2.bitwise_or(mask_final, component_mask)
            else:
                # Objeto grande: precisa de Watershed
                cnts, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    separated_object = MaskPostProcessor._apply_watershed_split(cnts[0], mask_filled.shape)
                    mask_final = cv2.bitwise_or(mask_final, separated_object)

        return mask_final

    def process(self, mask_prob_numpy, min_area=None):
        """
        Método de instância para ser chamado pelo inference.py.
        Corrigido erro de digitação de 'self.min_are' para 'self.min_area'.
        """
        area_to_use = min_area if min_area is not None else self.min_area
        
        return MaskPostProcessor.refine(
            mask_prob_numpy, 
            mask_prob_numpy.shape, 
            self.threshold, 
            area_to_use
        )