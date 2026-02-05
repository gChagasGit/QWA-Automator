import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max

class MaskPostProcessor:

    KERNEL_3x3 = np.ones((3, 3), np.uint8)

    def __init__(self, threshold=0.5, min_area=100):
        """
        Inicializador para compatibilidade com o app.py
        """
        self.threshold = threshold
        self.min_area = min_area

    @staticmethod
    def _is_touching_border(contour, shape):
        h, w = shape[:2]
        x, y, cw, ch = cv2.boundingRect(contour)
        return x <= 1 or y <= 1 or (x + cw) >= w - 1 or (y + ch) >= h - 1

    @staticmethod
    def _preprocess_mask(mask_prob_numpy, threshold):
        # Garante que é array numpy float ou uint8 normalizado
        mask_bin = (mask_prob_numpy > threshold).astype(np.uint8) * 255
        
        # Morfologia para limpar ruído
        mask_morph = cv2.erode(mask_bin, MaskPostProcessor.KERNEL_3x3, iterations=1)
        mask_morph = cv2.dilate(mask_morph, MaskPostProcessor.KERNEL_3x3, iterations=1)
        
        # Preenchimento de buracos (scipy)
        mask_filled_bool = ndimage.binary_fill_holes(mask_morph > 0)
        return (mask_filled_bool.astype(np.uint8) * 255)

    @staticmethod
    def _calculate_split_threshold(contours, shape, min_area):
        areas_validas = []
        for cnt in contours:
            if not MaskPostProcessor._is_touching_border(cnt, shape):
                area = cv2.contourArea(cnt)
                if area > min_area:
                    areas_validas.append(area)

        if not areas_validas:
            return float('inf')

        mean_area = np.mean(areas_validas)
        std_area = np.std(areas_validas)
        # Define que objetos muito maiores que a média + desvio padrão provavelmente são duplos
        return mean_area + std_area

    @staticmethod
    def _apply_watershed_split(contour, mask_shape):
        """
        Aplica Watershed para separar objetos aglutinados.
        """
        # 1. Isolar o objeto atual (ROI)
        mask_roi = np.zeros(mask_shape, dtype=np.uint8)
        cv2.drawContours(mask_roi, [contour], -1, 255, -1)

        # 2. Distance Transform & Picos
        dist = cv2.distanceTransform(mask_roi, cv2.DIST_L2, 5)
        # min_distance ajusta quão perto dois centros podem estar
        coords = peak_local_max(dist, min_distance=15, labels=mask_roi)

        if len(coords) < 2:
            return mask_roi # Nada para separar

        # 3. Criar Marcadores
        markers = np.zeros(mask_shape, dtype=np.int32)

        # A) Sementes (Foreground)
        for i, (r, c) in enumerate(coords):
            markers[r, c] = i + 2

        # B) Fundo (Background)
        sure_bg = cv2.dilate(mask_roi, MaskPostProcessor.KERNEL_3x3, iterations=3)
        markers[sure_bg == 0] = 1 

        # 4. Executar Watershed
        mask_roi_bgr = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(mask_roi_bgr, markers)

        # 5. Reconstrução
        mask_separated = np.zeros(mask_shape, dtype=np.uint8)
        mask_separated[markers > 1] = 255

        # Cria gap visual nas fronteiras (-1)
        cuts = (markers == -1).astype(np.uint8) * 255
        cuts_dilated = cv2.dilate(cuts, MaskPostProcessor.KERNEL_3x3, iterations=1)
        mask_separated[cuts_dilated > 0] = 0

        return mask_separated

    @staticmethod
    def refine(mask_prob_numpy, shape, threshold=0.5, min_area=100):
        """
        Método estático principal que executa todo o pipeline.
        """
        mask_filled = MaskPostProcessor._preprocess_mask(mask_prob_numpy, threshold)
        
        # Encontra contornos iniciais
        contours, _ = cv2.findContours(mask_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calcula limiar dinâmico de tamanho para decidir quem precisa de watershed
        threshold_split = MaskPostProcessor._calculate_split_threshold(contours, shape, min_area)

        mask_final = np.zeros_like(mask_filled)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue # Ignora ruído pequeno

            if area < threshold_split:
                # Objeto normal, desenha direto
                cv2.drawContours(mask_final, [cnt], -1, 255, -1)
            else:
                # Objeto gigante (provavelmente fundido), tenta separar
                separated_object = MaskPostProcessor._apply_watershed_split(cnt, mask_filled.shape)
                mask_final = cv2.bitwise_or(mask_final, separated_object)

        return mask_final

    def process(self, mask_prob_numpy):
        """
        Método de instância para ser chamado pelo inference.py
        """
        return MaskPostProcessor.refine(
            mask_prob_numpy, 
            mask_prob_numpy.shape, 
            self.threshold, 
            self.min_area
        )