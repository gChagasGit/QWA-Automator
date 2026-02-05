# yolo_unet3p_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

# ==============================================================================
# CLASSE 1: O ENCODER (Interna)
# ==============================================================================

class Encoder_YoloV8(nn.Module):
    """
    Extrator de características de um modelo YOLOv8. (Versão 4.0 - Auto-configurável)
    Esta versão adiciona o método `get_encoder_channels()` para descobrir
    dinamicamente o número de canais de saída do encoder.
    """
    def __init__(self, model_path: str):
        super().__init__()
        self.features = []
        self.model = YOLO(model_path, task='detect')
        self.feature_extractor = self.model.model
        self.encoder_channels_ = None # Cache

        for module in self.feature_extractor.modules():
            if isinstance(module, Detect):
                self.hook_handle = module.register_forward_hook(self.hook_fn)
                break
        else:
            print("AVISO: Camada 'Detect' não encontrada. "
                  "Se este for um modelo -seg (segmentação), isso é esperado. "
                  "Tentando encontrar a camada 'Segment'.")
            
            # Fallback para modelos -seg (Segment)
            try:
                from ultralytics.nn.modules.head import Segment
                for module in self.feature_extractor.modules():
                    if isinstance(module, Segment):
                        self.hook_handle = module.register_forward_hook(self.hook_fn)
                        print(f"Hook registrado com sucesso na camada 'Segment'.")
                        break
                else:
                    raise ValueError("Camada 'Detect' ou 'Segment' não encontrada.")
            except ImportError:
                 raise ValueError("Camada 'Detect' não encontrada e 'Segment' não pôde ser importada.")

        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def hook_fn(self, module, input, output):
        # A entrada para a camada 'Detect'/'Segment' é uma tupla
        self.features = list(input[0])

    def get_encoder_channels(self, dummy_input_size=(768, 1024)):
        if self.encoder_channels_ is not None:
            return self.encoder_channels_

        try:
            device = next(self.feature_extractor.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        H, W = dummy_input_size
        if H % 32 != 0 or W % 32 != 0:
             raise ValueError(f"dummy_input_size ({H}, {W}) deve ser múltiplo de 32.")
             
        dummy_input = torch.randn(1, 3, H, W).to(device)
        
        with torch.no_grad():
            _ = self(dummy_input) 
        
        if not self.features:
             raise RuntimeError("Falha ao capturar features na passagem dummy.")
        
        self.encoder_channels_ = [f.shape[1] for f in self.features]
        
        return self.encoder_channels_

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self.features = []
        _ = self.feature_extractor(x)
        
        if self.encoder_channels_ is None and self.features:
             self.encoder_channels_ = [f.shape[1] for f in self.features]

        return self.features

    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

# ==============================================================================
# CLASSE 2: O DECODER (Interna)
# ==============================================================================

class Decoder_UNet3Plus(nn.Module):
    """
    (Versão Final Limpa)
    Arquitetura de segmentação que combina o encoder YOLOv8
    com o decoder da UNet3+.
    """
    def __init__(self, encoder, n_classes=1, is_ds=False, dummy_input_size=(768, 1024)):
        super(Decoder_UNet3Plus, self).__init__()

        self.encoder = encoder
        self.is_ds = is_ds
        self.CatChannels = 64

        encoder_channels = encoder.get_encoder_channels(dummy_input_size=dummy_input_size)
        
        self.adapter_h3 = self.adapter_block(encoder_channels[0], self.CatChannels)
        self.adapter_h4 = self.adapter_block(encoder_channels[1], self.CatChannels)
        self.adapter_hd5 = self.adapter_block(encoder_channels[2], self.CatChannels)

        in_channels_hd4 = self.CatChannels * 3
        out_channels_hd4 = in_channels_hd4
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4d_1 = nn.Conv2d(in_channels_hd4, out_channels_hd4, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(out_channels_hd4)
        self.relu4d_1 = nn.ReLU(inplace=True)

        in_channels_hd3 = self.CatChannels + out_channels_hd4 + self.CatChannels
        out_channels_hd3 = in_channels_hd4
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.conv3d_1 = nn.Conv2d(in_channels_hd3, out_channels_hd3, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(out_channels_hd3)
        self.relu3d_1 = nn.ReLU(inplace=True)

        in_channels_hd2 = out_channels_hd3 + out_channels_hd4 + self.CatChannels
        out_channels_hd2 = in_channels_hd4
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.conv2d_1 = nn.Conv2d(in_channels_hd2, out_channels_hd2, 3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(out_channels_hd2)
        self.relu2d_1 = nn.ReLU(inplace=True)

        in_channels_hd1 = out_channels_hd2 + out_channels_hd3 + out_channels_hd4 + self.CatChannels
        out_channels_hd1 = 256
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.conv1d_1 = nn.Conv2d(in_channels_hd1, out_channels_hd1, 3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(out_channels_hd1)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.outconv1 = nn.Conv2d(out_channels_hd1, n_classes, 3, padding=1)

        if self.is_ds:
            self.dsout2 = nn.Conv2d(out_channels_hd2, n_classes, 1)
            self.dsout3 = nn.Conv2d(out_channels_hd3, n_classes, 1)
            self.dsout4 = nn.Conv2d(out_channels_hd4, n_classes, 1)

    def adapter_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        yolo_features = self.encoder(x)
        yolo_h3, yolo_h4, yolo_hd5 = yolo_features
        h3 = self.adapter_h3(yolo_h3)
        h4 = self.adapter_h4(yolo_h4)
        hd5 = self.adapter_hd5(yolo_hd5)
        h3_PT_hd4 = self.h3_PT_hd4(h3)
        h4_Cat_hd4 = h4
        hd5_UT_hd4 = self.hd5_UT_hd4(hd5)
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))
        h3_Cat_hd3 = h3
        hd4_UT_hd3 = self.hd4_UT_hd3(hd4)
        hd5_UT_hd3 = self.hd5_UT_hd3(hd5)
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))
        hd3_UT_hd2 = self.hd3_UT_hd2(hd3)
        hd4_UT_hd2 = self.hd4_UT_hd2(hd4)
        hd5_UT_hd2 = self.hd5_UT_hd2(hd5)
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))
        hd2_UT_hd1 = self.hd2_UT_hd1(hd2)
        hd3_UT_hd1 = self.hd3_UT_hd1(hd3)
        hd4_UT_hd1 = self.hd4_UT_hd1(hd4)
        hd5_UT_hd1 = self.hd5_UT_hd1(hd5)
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))
        d1 = F.interpolate(self.outconv1(hd1), size=x.shape[2:], mode='bilinear', align_corners=False)
        if self.is_ds:
            d2 = F.interpolate(self.dsout2(hd2), size=x.shape[2:], mode='bilinear', align_corners=False)
            d3 = F.interpolate(self.dsout3(hd3), size=x.shape[2:], mode='bilinear', align_corners=False)
            d4 = F.interpolate(self.dsout4(hd4), size=x.shape[2:], mode='bilinear', align_corners=False)
            return [d1, d2, d3, d4]
        else:
            return d1

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            if module is not self.encoder:
                module.train(mode)
        return self

    def eval(self):
        return self.train(False)

# ==============================================================================
# CLASSE 3: O MODELO HÍBRIDO ENCÁPSULADO (PARA IMPORTAÇÃO)
# ==============================================================================

class YoloV8_Unet3p(nn.Module):
    """
    Encapsula o Encoder_YoloV8 e o Decoder_UNet3Plus em uma
    única classe para facilitar a inicialização, treino e inferência.
    """
    def __init__(self, 
                 yolo_model_path: str, 
                 n_classes: int = 1, 
                 is_ds: bool = False, 
                 dummy_input_size: tuple = (768, 1024),
                 decoder_weights_path: str = None): # <--- NOVO ARGUMENTO
        
        super().__init__()
        
        # 1. Cria o Encoder
        print(f"🔄 Carregando pesos do encoder de: {yolo_model_path}")
        self.encoder = Encoder_YoloV8(yolo_model_path) 
        
        # 2. Cria o Decoder
        self.decoder = Decoder_UNet3Plus(
            encoder=self.encoder,
            n_classes=n_classes,
            is_ds=is_ds,
            dummy_input_size=dummy_input_size
        )

        # 3. Carregamento Automático de Pesos (Se o caminho for fornecido)
        if decoder_weights_path:
            self.load_decoder_weights(decoder_weights_path)

    def forward(self, x: torch.Tensor):
        return self.decoder(x)

    def train(self, mode=True):
        self.training = mode
        self.decoder.train(mode) 
        return self

    def eval(self):
        return self.train(False)

    def load_decoder_weights(self, path: str):
        """
        Carrega os pesos de forma robusta, lidando com checkpoints do Ignite
        ou dicionários de estado puros.
        """
        print(f"🔄 Carregando pesos do decoder de: {path}")
        try:
            # Carrega para CPU primeiro para evitar erros de dispositivo antes do .to(device)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            
            # Lógica para extrair o state_dict correto
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Tenta carregar no modelo inteiro (self)
            # O strict=False é crucial aqui pois os pesos do encoder (YOLO) 
            # já foram carregados na inicialização e podem não estar neste checkpoint,
            # ou vice-versa.
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            
            print("✅ Pesos carregados com sucesso (strict=False).")
            # Opcional: imprimir o que faltou para debug, se necessário
            # if len(missing) > 0: print(f"   Chaves ausentes (normal para encoder congelado): {len(missing)}")
            
        except Exception as e:
            print(f"❌ AVISO: Falha ao carregar checkpoint: {e}")
            print("   O treinamento continuará com pesos inicializados aleatoriamente (exceto YOLO).")
