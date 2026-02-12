import io
import os
import sys
import cv2
import yaml
import shutil

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# --- 1. CONFIGURAÇÃO DE PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)

# --- 2. IMPORTS ---
try:
    from src.core.inference_onnx import ONNXModel, run_inference
    from src.core.metrics import calculate_qwa_metrics, calculate_area_scale_factor
    from src.core.post_processing import MaskPostProcessor
    from src.gui.visualization import draw_grid
except ImportError as e:
    st.error(f"Erro Crítico de Importação: {e}")
    st.stop()

# Carrega as configurações dos modelos
config_path = os.path.join(root_dir, "config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config_models = yaml.safe_load(f)

# Lista de nomes de modelos disponíveis
model_options = list(config_models['models'].keys())

# --- 3. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="QWA Automator", layout="wide", page_icon="⚡")

# --- CSS ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 25%; max-width: 25%; }
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; margin-top: 0rem; }
    h1 { padding-top: 0rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

TEMP_DIR = os.path.join(root_dir, "temp_masks")
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

# --- 4. FUNÇÕES DE UTILIDADE ---

@st.cache_resource
def load_onnx_model(model_key):
    try:
        # Buscamos os dados direto do nosso dicionário de configuração
        m_info = config_models['models'][model_key]
        
        # O caminho do modelo deve ser relativo à raiz do projeto
        onnx_path = os.path.join(root_dir, m_info['path'])
        
        if not onnx_path or not os.path.exists(onnx_path):
            return None, "Arquivo do modelo não encontrado."
        
        adapter = ONNXModel(
            onnx_path=onnx_path,
            mean=m_info['mean'],
            std=m_info['std'],
            input_size=m_info['input_size']
        )
        return adapter, None
    except Exception as e:
        return None, f"Erro ao carregar ONNX: {str(e)}"

def filter_border_vessels(df, apenas_inside):
    if df is None or df.empty: return df
    if apenas_inside and 'Inside' in df.columns:
        return df[df['Inside'] == True]
    return df

def calcular_resumo_imagem(df_vessels, filename, img_area_mm2, total_pixels):
    if df_vessels is None or df_vessels.empty: return None
    n = len(df_vessels)
    
    porosidade = (df_vessels['Area_px'].sum() / total_pixels) * 100
    return {
        "Arquivo": filename, 
        "Nº Vasos": n, 
        "Freq. (v/mm²)": n / img_area_mm2,
        "Média do Ø Maior (µm)": df_vessels['Major_Axis_um'].mean(),
        "Média do Ø Menor (µm)": df_vessels['Minor_Axis_um'].mean(),
        "Área Média (µm²)": df_vessels['Area_um2'].mean(),
        "Porosidade (%)": porosidade
    }

def calcular_linhas_resumo(df_base, label_col_name="Arquivo"):
    if df_base.empty: return pd.DataFrame(), pd.DataFrame()
    cols_num = df_base.select_dtypes(include=[np.number]).columns
    row_sum = df_base[cols_num].sum(); row_sum[label_col_name] = "TOTAL"
    row_mean = df_base[cols_num].mean(); row_mean[label_col_name] = "MÉDIA"
    return pd.DataFrame([row_sum]), pd.DataFrame([row_mean])

def agrupar_por_quadrante(df_imagem, img_area_mm2):
    area_q0 = max(img_area_mm2 - 4.0, 0.001)
    mapa_areas = {0: area_q0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    linhas = []
    for q in range(5):
        df_q = df_imagem[df_imagem['Quadrante'] == q]
        roi_mm2 = mapa_areas.get(q, 1.0)
        roi_um2 = roi_mm2 * 1_000_000
        n = len(df_q)
        if n > 0:
            porosidade_local = (df_q['Area_um2'].sum() / roi_um2) * 100
            stats = {
                "Quadrante": str(q), "Nº Vasos": n, "Freq. (v/mm²)": n / roi_mm2,
                "Média do Ø Maior": df_q['Major_Axis_um'].mean(), "Ø Maior Std": df_q['Major_Axis_um'].std(),
                "Média do Ø Menor": df_q['Minor_Axis_um'].mean(), "Ø Menor Std": df_q['Minor_Axis_um'].std(),
                "Área Média": df_q['Area_um2'].mean(), "Área Std": df_q['Area_um2'].std(),
                "Porosidade (%)": porosidade_local
            }
        else:
            stats = {"Quadrante": str(q), "Nº Vasos": 0, "Freq. (v/mm²)": 0.0, "Média do Ø Maior": 0.0, "Ø Maior Std": 0.0, "Média do Ø Menor": 0.0, "Ø Menor Std": 0.0, "Área Média": 0.0, "Área Std": 0.0, "Porosidade (%)": 0.0}
        linhas.append(stats)
    
    df = pd.DataFrame(linhas)
    df_sum, df_mean = calcular_linhas_resumo(df, label_col_name="Quadrante")
    dfs_to_concat = [d.dropna(axis=1, how='all') for d in [df, df_sum, df_mean] if not d.empty and not d.isna().all().all()]
    return pd.concat(dfs_to_concat, ignore_index=True).convert_dtypes() if dfs_to_concat else df

st.header("📂 Entrada de Dados")

uploaded_files = st.file_uploader(
    "Selecione a pasta de imagens", 
    accept_multiple_files="directory",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} imagens carregadas.")

# --- 5. SIDEBAR: CONFIGURAÇÕES E ENTRADA ---
with st.sidebar:
    st.header("⚙️ Calibração")
    resolution = st.number_input("Resolução (µm/pixel):", min_value=0.01, value=1.0638, format="%.4f", help="Razão da Área da imagem em micrômetros pela área em pixels.")
    min_area_um2 = st.number_input("Área Mínima do vaso (µm²):", value=1000.0, step=100.0, help="Vasos com área menor que este valor serão ignorados na análise.")
    
    st.divider()
    st.header("🔍 Segmentação")
    
    # Seleção do Modelo
    selected_model_name = st.selectbox(
        "Selecione o Modelo ONNX",
        options=model_options,
        index=model_options.index(config_models.get('active_model', model_options[0]))
    )
    
    # Obtém os parâmetros do modelo selecionado
    m_cfg = config_models['models'][selected_model_name]
    
    # Cálculo do total em pixels das máscara que o modelo gera.
    img_total_px=(m_cfg['input_size'][0] * m_cfg['input_size'][1])
    
    # Exibe informações do modelo (opcional, para transparência científica)
    st.caption(f"Resolução de treino: {m_cfg['input_size'][0]}x{m_cfg['input_size'][1]}")
    
    threshold_model = st.slider("Confiança:", 0.1, 0.9, 0.5)
    edge_truncated_vessels = st.checkbox("Descartar vasos seccionados pela borda?", value=False)

# --- 6. INICIALIZAÇÃO DE ESTADO ---
for key in ['results_raw', 'uploaded_files_map', 'pixel_size_map', 'processado']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'results_raw' else ({} if 'map' in key else False)

# --- 7. PROCESSAMENTO ---
if st.button("🚀 Gerar Relatório", width="stretch", type="primary"):
    if not uploaded_files:
        st.error("Nenhuma imagem para processar.")
    else:
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR); os.makedirs(TEMP_DIR)
        st.session_state['results_raw'] = []
        st.session_state['uploaded_files_map'] = {} 
        
        with st.spinner("Processando..."):
            adapter, erro = load_onnx_model(selected_model_name)
            if not adapter: st.error(erro); st.stop()
            
            bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                # Extrai apenas o nome do arquivo (ex: '6.jpeg') e ignora o caminho da pasta
                filename = os.path.basename(file.name) 
                try:
                    # Garantimos que lemos o buffer do início
                    file.seek(0)
                    file_bytes = file.read()
                    img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                    
                    orig_w, orig_h = img_pil.size
                    
                    fator_escala = calculate_area_scale_factor(img_pil.size, adapter.input_size)
                    min_area_scaled = int(round((1/fator_escala) * (min_area_um2 / (resolution**2))))
                    
                    post_proc = MaskPostProcessor(threshold=threshold_model, min_area=min_area_scaled)
                    mask_array = run_inference(adapter, img_pil, post_proc)
                    
                    temp_path = os.path.join(TEMP_DIR, f"temp_{filename}.png")
                    Image.fromarray(mask_array).save(temp_path)
                    
                    df_img = calculate_qwa_metrics(temp_path, img_pil.size, resolution)
                    
                    if df_img is not None and not df_img.empty:
                        df_img.insert(0, 'Arquivo', filename)
                        df_img['Img_Area_mm2'] = ((orig_w * orig_h) * (resolution ** 2)) / 1_000_000.0
                        st.session_state['results_raw'].append(df_img)
                        st.session_state['uploaded_files_map'][filename] = file
                        st.session_state['pixel_size_map'][filename] = resolution
                except Exception as e:
                    st.error(f"Erro em {filename}: {e}")
                bar.progress((i+1)/len(uploaded_files))
        
        st.session_state['processado'] = True
        del adapter
        st.rerun()

# --- EXIBIÇÃO DOS RESULTADOS ---
if st.session_state['processado']:
    summary_list = []
    for df_raw in st.session_state['results_raw']:
        df_filtered = filter_border_vessels(df_raw, edge_truncated_vessels)
        if not df_filtered.empty:
            filename = df_raw['Arquivo'].iloc[0]
            area_mm2 = df_raw['Img_Area_mm2'].iloc[0]
            stats = calcular_resumo_imagem(df_filtered, filename, area_mm2, img_total_px)
            if stats: summary_list.append(stats)
    
    if not summary_list:
        st.warning("⚠️ **Nenhum vaso foi detectado nas imagens processadas.**")
        st.info("""
            **Sugestões para ajuste:**
            1. **Revise a Imagem:** Verifique se a qualidade ou o contraste da captura original permite a identificação dos vasos.
            2. **Ajuste o Filtro:** A 'Área Mínima (µm²)' pode estar muito alta, excluindo todos os vasos detectados.
            3. **Troque o Modelo:** O modelo de IA selecionado pode não ser o mais adequado para este tipo específico de madeira.
        """)
    else:
        df_summary = pd.DataFrame(summary_list)
        st.header("📋 Resumo Global")
        df_base = df_summary[["Arquivo", "Nº Vasos", "Freq. (v/mm²)", "Média do Ø Maior (µm)", "Média do Ø Menor (µm)", "Área Média (µm²)", "Porosidade (%)"]]
        df_s, df_m = calcular_linhas_resumo(df_base, label_col_name="Arquivo")
        dfs = [d.dropna(axis=1, how='all') for d in [df_base, df_s, df_m] if not d.empty]
        
        df_final = pd.concat(dfs, ignore_index=True).convert_dtypes()
        height_global = 600 if len(df_final) > 25 else "content"
        
        st.dataframe(
            df_final.style.format(precision=2, na_rep="-").apply(
                lambda x: ['background-color: #e6e9ef; color: #292933' if x['Arquivo'] in ['TOTAL', 'MÉDIA'] else '' for i in x], axis=1),
            width="stretch", 
            hide_index=True,
            height=height_global)

        st.divider()
        st.header("🔍 Detalhes por Amostra")
        selected_file = st.selectbox("Selecione:", options=df_summary['Arquivo'].unique())
        
        if selected_file:
            exibir_visualizacao = st.checkbox("📸 Exibir Imagem e Máscara (Segmentação)", value=False)
            
            df_raw_sel = [d for d in st.session_state['results_raw'] if d['Arquivo'].iloc[0] == selected_file][0]
            df_filt_sel = filter_border_vessels(df_raw_sel, edge_truncated_vessels)            
            
            st.markdown("""
                <style>
                .stImage > img { padding: 10px; background-color: #f0f2f6; border-radius: 10px; }
                </style>
                """, unsafe_allow_html=True)

            if exibir_visualizacao:
                col1, col2 = st.columns(2, gap="large")
                
                with col1:
                    file_ref = st.session_state['uploaded_files_map'][selected_file]
                    file_ref.seek(0) # Volta para o início do arquivo
                    img_full = Image.open(file_ref).convert("RGB")
                    st.image(img_full, caption=f"Original: {selected_file}", width="stretch")
                    
                with col2:
                    mask_path = os.path.join(TEMP_DIR, f"temp_{selected_file}.png")
                    mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                    mask_viz = draw_grid(
                        mask_cv,
                        img_full.size, 
                        st.session_state['pixel_size_map'][selected_file], 
                        df_filt_sel
                    )
                    
                    st.image(mask_viz, caption="Segmentação + Quadrantes (1mm²)", width="stretch")
                    st.caption("**Quadrantes:** :blue[1-Azul] | :green[2-Verde] | :red[3-Vermelho] | :orange[4-Laranja]")
  
            st.subheader("📊 Estatísticas por Quadrante")
            df_q = agrupar_por_quadrante(df_filt_sel, df_raw_sel['Img_Area_mm2'].iloc[0])
            st.dataframe(
                df_q.style.format(precision=2, na_rep="-").apply(
                    lambda x: ['background-color: #e6e9ef; color: #292933' if x['Quadrante'] in ['TOTAL', 'MÉDIA'] else '' for i in x], axis=1),
                width="stretch", hide_index=True) 

            # --- DOWNLOADS ---
            st.divider()
            final_out_dir = os.path.join(root_dir, "data/output_results")
            if not os.path.exists(final_out_dir): os.makedirs(final_out_dir)
            
            csv_sum = os.path.join(final_out_dir, "resumo_anatomico.csv")
            df_summary.to_csv(csv_sum, sep=';', encoding='utf-8-sig', index=False)
            
            df_all_raw = pd.concat(st.session_state['results_raw'], ignore_index=True)
            if "Imagem" in df_all_raw.columns: df_all_raw.drop(columns=["Imagem"], inplace=True)
                
            csv_raw = os.path.join(final_out_dir, "dados_brutos_vasos.csv")
            df_all_raw.to_csv(csv_raw, sep=';', encoding='utf-8-sig', index=False)

            c_dl1, c_spacer, c_dl2 = st.columns([2, 4, 2])
            with c_dl1:
                with open(csv_sum, "rb") as f: 
                    st.download_button("📥 Baixar Resumo (CSV)", f, "resumo_anatomico.csv", width="stretch")
            with c_dl2:
                with open(csv_raw, "rb") as f: 
                    st.download_button("📥 Dados Brutos (Completo)", f, "dados_brutos_vasos.csv", width="stretch")

# CORREÇÃO DAS VERIFICAÇÕES FINAIS:
elif uploaded_files: # <-- Mudança de files_to_process para uploaded_files
    st.info("Clique em 'Gerar Relatório' para iniciar.")
else:
    st.info("Faça upload das imagens para começar.")