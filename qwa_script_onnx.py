# -*- coding: utf-8 -*-
import argparse
import os
import sys
import pandas as pd
import shutil
import yaml
from PIL import Image
from tqdm import tqdm

# Garante que o Python encontre os módulos da pasta src/
sys.path.append(os.getcwd())
print(f"🔍 Diretório atual: {os.getcwd()}")

# Tenta importar os módulos do projeto
try:
    from src.core.metrics import calculate_qwa_metrics, calculate_area_scale_factor
    # Importa o adaptador ONNX específico
    from src.core.inference_onnx import ONNXModel, run_inference
    from src.core.post_processing import MaskPostProcessor
    print("✅ Bibliotecas e Módulos ONNX importados com sucesso!")
except ImportError as e:
    print(f"⚠️ Erro de importação: {e}")
    print("Certifique-se de estar rodando o script na raiz do projeto (QWA_Automator_V1).")
    sys.exit(1)

# --- FUNÇÕES AUXILIARES (Idênticas ao script original) ---
def filter_border_vessels(df, apenas_inside):
    if apenas_inside and 'Inside' in df.columns:
        return df[df['Inside'] == True]
    return df

def calculate_summary_image(df_vessels, filename, img_area_mm2, img_total_px=(640*640)):
    if df_vessels is None or df_vessels.empty: return None
    n = len(df_vessels)
    
    porosity = (df_vessels['Area_px'].sum() / img_total_px) * 100
    
    return {
        "Arquivo": filename, 
        "Nº Vasos": n, 
        "Freq. (v/mm²)": n / img_area_mm2,
        "Ø Maior Médio (µm)": df_vessels['Major_Axis_um'].mean(),
        "Ø Maior Std": df_vessels['Major_Axis_um'].std(),
        "Ø Menor Médio (µm)": df_vessels['Minor_Axis_um'].mean(),
        "Ø Menor Std": df_vessels['Minor_Axis_um'].std(),
        "Área Média (µm²)": df_vessels['Area_um2'].mean(),
        "Área Std": df_vessels['Area_um2'].std(),
        "Porosidade (%)": porosity
    }

def load_model_onnx(onnx_path, mean, std, input_size):
    """
    Inicializa o adaptador ONNX.
    O próprio adaptador gerencia Providers (CPU/OpenVINO/CUDA) internamente.
    """
    if not os.path.exists(onnx_path):
        print(f"❌ Modelo não encontrado em: {onnx_path}")
        sys.exit(1)
        
    try:
        adapter = ONNXModel(onnx_path, mean, std, input_size)
        return adapter
    except Exception as e:
        print(f"❌ Erro fatal ao carregar modelo ONNX: {e}")
        sys.exit(1)

def create_default_config(filename="config_onnx.yaml"):
    """Cria um arquivo de configuração padrão se ele não existir."""
    default_yaml = """paths:
  input: "input_images"  # Coloque suas imagens aqui. Exemplo: "data/input_images"
  output: "output_results" # Resultados serão salvos aqui. Exemplo: "data/output_results"

parameters:
  resolution_um_px: 1.0638  # Resolução em micrometros por pixel
  min_area_px: 1000          # Área mínima para considerar um vaso
  threshold: 0.5            # Confiança da IA (IoU threshold ou Score)
  ignore_border: false      # Se true, ignora vasos cortados na borda
  save_masks: true          # Salvar as máscaras geradas?

active_model: "InsideWood-IW" # Nome do modelo a ser carregado (deve existir na seção models)

models:
  InsideWood-IW:    # Modelo de Segmentação selecionado em active_model
    path: "model/insidewood_segmenter.onnx" # Caminho para o modelo ONNX
    mean: [0.6187, 0.5177, 0.5508] # Média de normalização (RGB) do modelo
    std: [0.2334, 0.2704, 0.2530]  # Desvio padrão de normalização (RGB) do modelo
    input_size: [640, 640] # Tamanho de entrada esperado pelo modelo (W, H)
"""
    try:
        with open(filename, "w") as f:
            f.write(default_yaml.strip())
        print(f"✅ Arquivo de configuração padrão criado: '{filename}'")
    except Exception as e:
        print(f"⚠️ Não foi possível criar o arquivo de config: {e}")

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Processamento Batch QWA (Versão ONNX)")
    parser.add_argument('config_file', type=str, nargs='?', default='config.yaml', 
                        help='Caminho para o arquivo .yaml de configuração')

    args = parser.parse_args()

    # 1. Carregar Configuração
    if not os.path.exists(args.config_file):
        print(f"⚠️ Config '{args.config_file}' não encontrada. Criando padrão...")
        create_default_config(args.config_file)

    try:
        with open(args.config_file, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Erro ao ler arquivo YAML: {e}")
        sys.exit(1)

    # Extrair variáveis do YAML
    try:
        
        input_dir = cfg['paths']['input']
        output_dir = cfg['paths']['output']
        
        resolution = cfg['parameters'].get('resolution_um_px', 1.0638)
        min_area_um = cfg['parameters'].get('min_area_um', 1000)
        threshold_model = cfg['parameters'].get('threshold', 0.5)
        ignore_border = cfg['parameters'].get('ignore_border', False)
        save_masks = cfg['parameters'].get('save_masks', False)
    
        # ============================================================
        
        active_model_name = cfg.get('active_model', 'vessel')
        m_cfg = cfg['models'][active_model_name]
        
    except KeyError as e:
        print(f"❌ Campo obrigatório faltando no YAML: {e}")
        sys.exit(1)
        
    # Cálculo do min_area_obj em pixels para o MaskPostProcessor.
    min_area_obj = int(round(min_area_um / (resolution ** 2)))

    # Cálculo do total em pixels das máscara que o modelo gera.
    img_total_px=(m_cfg['input_size'][0] * m_cfg['input_size'][1])
    
    # 2. Inicializar Modelo
    print(f"🚀 Carregando o modelo {m_cfg['path']}")
    adapter = load_model_onnx(
        onnx_path=m_cfg['path'],
        mean=m_cfg['mean'],
        std=m_cfg['std'],
        input_size=m_cfg['input_size']
    )
    print(f"✅ Modelo carregado. Provider: {adapter.provider}")

    # Verificar diretórios
    if not os.path.exists(input_dir):
        print(f"❌ Diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
    if save_masks:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    # Pasta temporária para cálculos intermediários
    root_dir = os.getcwd()
    temp_dir = os.path.join(root_dir, "temp_batch")
    os.makedirs(temp_dir, exist_ok=True)
    
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(exts)])

    if not files:
        print(f"❌ Nenhuma imagem válida encontrada em '{input_dir}'.")
        return

    results_raw = []
    summary_list = []

    print(f"📂 Processando {len(files)} imagens de: {input_dir}")
    pbar = tqdm(files, unit="img")
    
    for filename in pbar:
        pbar.set_description(f"Processando {filename}")
        file_path = os.path.join(input_dir, filename)

        try:
            # Carregar imagem
            img_pil = Image.open(file_path).convert("RGB")
            orig_w, orig_h = img_pil.size

            # Cálculo do fator de escala e min_area dinâmico
            fator_escala = calculate_area_scale_factor(img_pil.size, adapter.input_size)
            min_area_scaled = int(round((1/fator_escala) * min_area_obj))
            
            # Processador atualizado (Post-processing com regionprops)
            post_proc = MaskPostProcessor(threshold=threshold_model, min_area=min_area_scaled)
            
            mask_array = run_inference(adapter, img_pil, post_proc)

            # Salvar máscara temporária para cálculo de métricas (reaproveitando lógica existente)
            temp_path = os.path.join(temp_dir, f"temp_{filename}.png")
            Image.fromarray(mask_array).save(temp_path)

            # Cálculos QWA
            img_area_mm2 = ((orig_w * orig_h) * (resolution ** 2)) / 1_000_000.0
            
            # Chama a função de métricas do core
            df_img = calculate_qwa_metrics(temp_path, img_pil.size, resolution)

            if df_img is not None and not df_img.empty:
                df_img.insert(0, 'Arquivo', filename)
                df_img['Img_Area_mm2'] = img_area_mm2
                results_raw.append(df_img)

                # Estatísticas sumarizadas
                df_filtered = filter_border_vessels(df_img, ignore_border)
                stats = calculate_summary_image(df_filtered, filename, img_area_mm2, img_total_px)
                if stats: summary_list.append(stats)

                # --- Salvamento da Máscara com Redimensionamento ---
                if save_masks:
                    output_mask_path = os.path.join(output_dir, "masks", f"mask_{os.path.splitext(filename)[0]}.png")
                    
                    # Em vez de shutil.copy, redimensionamos para a dimensão original
                    # mask_array contém a saída do post_processing (0 ou 255)
                    mask_pil = Image.fromarray(mask_array)
                    
                    # Redimensiona de volta para (orig_w, orig_h) usando vizinho mais próximo 
                    # para não criar valores intermediários na binarização
                    mask_final_resized = mask_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
                    
                    mask_final_resized.save(output_mask_path)

            # Limpeza do arquivo temporário
            if os.path.exists(temp_path): os.remove(temp_path)

        except Exception as e:
            tqdm.write(f"⚠️ Erro em {filename}: {e}")

    # 4. Salvar Resultados Finais
    print("\n💾 Salvando planilhas...")
    if summary_list:
        summary_path = os.path.join(output_dir, "resumo.csv")
        pd.DataFrame(summary_list).to_csv(summary_path, sep=';', encoding='utf-8-sig', index=False)
        print(f"   -> {summary_path}")

    if results_raw:
        df_all = pd.concat(results_raw, ignore_index=True)
        if "Imagem" in df_all.columns: df_all.drop(columns=["Imagem"], inplace=True)
        
        raw_path = os.path.join(output_dir, "dados_brutos.csv")
        df_all.to_csv(raw_path, sep=';', encoding='utf-8-sig', index=False)
        print(f"   -> {raw_path}")
    else:
        print("⚠️ Nenhum vaso detectado em nenhuma imagem.")

    # Limpeza final
    try: shutil.rmtree(temp_dir)
    except: pass
    
    print("🏁 Processamento Concluído.")

if __name__ == "__main__":
    main()