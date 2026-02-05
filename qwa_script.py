import argparse
import os
import sys
import pandas as pd
import torch
import shutil
import yaml
from PIL import Image
from tqdm import tqdm

current_dir = os.getcwd()

try:
    from model.yolo_unet3p_model import YoloV8_Unet3p
    from src.core.metrics import calculate_qwa_metrics
    from src.core.inference import run_inference, LocalModelAdapter
    from src.core.post_processing import MaskPostProcessor
    print("✅ Bibliotecas e Módulos importados com sucesso!")
except ImportError as e:
    print(f"⚠️ Erro de importação: {e}")
    print("Certifique-se de que as pastas 'model' e 'src' estão na raiz do Colab.")

# --- FUNÇÕES AUXILIARES ---
def filtrar_vasos(df, apenas_inside):
    if apenas_inside and 'Inside' in df.columns:
        return df[df['Inside'] == True]
    return df

def calcular_resumo_imagem(df_vessels, filename, img_area_mm2):
    if df_vessels is None or df_vessels.empty: return None
    n = len(df_vessels)
    porosidade = (df_vessels['Area_px'].sum() / (1024 * 768)) * 100
    return {
        "Arquivo": filename, "Nº Vasos": n, "Freq. (v/mm²)": n / img_area_mm2,
        "Ø Maior Médio (µm)": df_vessels['Major_Axis_um'].mean(),
        "Ø Maior Std": df_vessels['Major_Axis_um'].std(),
        "Ø Menor Médio (µm)": df_vessels['Minor_Axis_um'].mean(),
        "Ø Menor Std": df_vessels['Minor_Axis_um'].std(),
        "Área Média (µm²)": df_vessels['Area_um2'].mean(),
        "Área Std": df_vessels['Area_um2'].std(),
        "Porosidade (%)": porosidade
    }

def carregar_modelo(yolo_path, unet_path, device):
    try:
        model = YoloV8_Unet3p(
            n_classes=1, is_ds=False, dummy_input_size=(768, 1024),
            yolo_model_path=yolo_path, decoder_weights_path=unet_path
        ).to(device)
        model.eval()
        return LocalModelAdapter(model, device)
    except Exception as e:
        print(f"❌ Erro fatal ao carregar modelo: {e}")
        sys.exit(1)

def create_config_path():
    config_path = "config.yaml"

    # OBS: O conteúdo abaixo deve ficar encostado na margem esquerda
    # para evitar erros de leitura do YAML.
    default_yaml = """paths:
  input: "/app/host/data/input_images"
  output: "/app/host/data/output_results"
  yolo_model: "model/model_yolo_detect.pt"
  unet_model: "model/model_unet3p_segment.pt"

parameters:
  resolution_um_px: 1.0638  # Resolução em micrometros por pixel
  min_area_px: 100          # Área mínima para considerar um vaso
  threshold: 0.5            # Confiança da IA
  ignore_border: false      # Se true, ignora vasos cortados na borda (Inside=False)
  save_masks: false          # Se true, salva as máscaras PNG
"""

    # Cria o arquivo
    with open(config_path, "w") as f:
        f.write(default_yaml.strip()) # .strip() remove a primeira quebra de linha vazia

    print(f"✅ Arquivo '{config_path}' criado com sucesso!")

# Define root_dir para execução no notebook (usado para localizar pastas temporárias)
root_dir = os.getcwd()

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Processamento Batch via YAML")
    parser.add_argument('config_file', type=str, help='Caminho para o arquivo config.yaml')

    # No notebook, evitamos erro de parser chamando args manualmente se necessário
    try:
        args = parser.parse_args()
    except:
        # Fallback para usar o config.yaml padrão se executado interativamente
        args = parser.parse_args(['config.yaml'])

    # 1. Ler YAML
    if not os.path.exists(args.config_file): create_config_path()

    with open(args.config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # Extrair configurações
    input_dir = cfg['paths']['input']
    output_dir = cfg['paths']['output']
    yolo_path = cfg['paths']['yolo_model']
    unet_path = cfg['paths']['unet_model']

    resolution = cfg['parameters']['resolution_um_px']
    min_area = cfg['parameters']['min_area_px']
    threshold = cfg['parameters']['threshold']
    ignore_border = cfg['parameters']['ignore_border']
    save_masks = cfg['parameters']['save_masks']

    # 2. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando QWA Batch (YAML) em: {device}, {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if not os.path.exists(input_dir):
        print(f"❌ Input não encontrado: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    if save_masks:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    temp_dir = os.path.join(root_dir, "temp_batch_yaml")
    os.makedirs(temp_dir, exist_ok=True)

    # 3. Execução
    adapter = carregar_modelo(yolo_path, unet_path, device)
    post_proc = MaskPostProcessor(threshold=threshold, min_area=min_area)

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(exts)])

    if not files:
        print("❌ Nenhuma imagem válida encontrada.")
        return

    results_raw = []
    summary_list = []

    pbar = tqdm(files, unit="img")
    for filename in pbar:
        pbar.set_description(f"Processando {filename}")
        file_path = os.path.join(input_dir, filename)

        try:
            img_pil = Image.open(file_path).convert("RGB")
            orig_w, orig_h = img_pil.size

            mask_array = run_inference(adapter, img_pil, post_proc, threshold)

            temp_path = os.path.join(temp_dir, f"temp_{filename}.png")
            Image.fromarray(mask_array).save(temp_path)

            img_area_mm2 = ((orig_w * orig_h) * (resolution ** 2)) / 1_000_000.0

            df_img = calculate_qwa_metrics(temp_path, orig_w, orig_h, resolution)

            if df_img is not None and not df_img.empty:
                df_img.insert(0, 'Arquivo', filename)
                df_img['Img_Area_mm2'] = img_area_mm2

                results_raw.append(df_img)

                df_filtered = filtrar_vasos(df_img, ignore_border)
                stats = calcular_resumo_imagem(df_filtered, filename, img_area_mm2)
                if stats: summary_list.append(stats)

                if save_masks:
                    shutil.copy(temp_path, os.path.join(output_dir, "masks", f"mask_{os.path.splitext(filename)[0]}.png"))

            if os.path.exists(temp_path): os.remove(temp_path)

        except Exception as e:
            tqdm.write(f"⚠️ Erro em {filename}: {e}")

    # 4. Salvar
    print("\n💾 Salvando resultados...")
    if summary_list:
        pd.DataFrame(summary_list).to_csv(os.path.join(output_dir, "resumo.csv"), sep=';', encoding='utf-8-sig', index=False)

    if results_raw:
        df_all = pd.concat(results_raw, ignore_index=True)
        if "Imagem" in df_all.columns: df_all.drop(columns=["Imagem"], inplace=True)
        df_all.to_csv(os.path.join(output_dir, "dados_brutos.csv"), sep=';', encoding='utf-8-sig', index=False)

    try: os.rmdir(temp_dir)
    except: pass
    print("🏁 Concluído.")

if __name__ == "__main__":
    main()