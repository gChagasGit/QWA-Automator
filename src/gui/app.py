import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
import shutil
import cv2

# --- 1. CONFIGURAÇÃO DE PATHS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)

# --- 2. IMPORTS (Versão ONNX) ---
try:
    # Importa os módulos específicos para ONNX
    from src.core.inference_onnx import ONNXModelAdapter, run_inference
    from src.core.metrics import calculate_qwa_metrics, calculate_area_scale_factor
    from src.core.post_processing import MaskPostProcessor
    from src.gui.visualization import desenhar_grid_quadrantes
except ImportError as e:
    st.error(f"Erro Crítico de Importação: {e}")
    st.info("Verifique se o arquivo src/core/inference_onnx.py existe e se as dependências estão instaladas.")
    st.stop()

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

def listar_modelos_onnx(diretorio):
    """Lista apenas arquivos .onnx na pasta model"""
    if not os.path.exists(diretorio): return []
    arquivos = [f for f in os.listdir(diretorio) if f.lower().endswith('.onnx')]
    return sorted(arquivos)

@st.cache_resource
def load_onnx_model(path_model):
    """Carrega o modelo ONNX e cacheia o adaptador na memória"""
    if not path_model or not os.path.exists(path_model):
        return None, "Arquivo do modelo não encontrado."

    try:
        # O adaptador ONNX gerencia a sessão e providers (CPU/OpenVINO) automaticamente
        adapter = ONNXModelAdapter(path_model)
        
        # Log discreto do provider em uso (CPUExecutionProvider ou OpenVINO/CUDA)
        providers = adapter.session.get_providers()
        print(f"Modelo carregado. Engine: {providers[0]}")
        
        return adapter, None
    except Exception as e:
        return None, f"Erro ao carregar ONNX: {str(e)}"

# --- LÓGICA DE CÁLCULO (Mantida idêntica para consistência) ---
def filtrar_vasos(df, apenas_inside):
    if apenas_inside and 'Inside' in df.columns:
        return df[df['Inside'] == True]
    return df

def calcular_resumo_imagem(df_vessels, filename, img_area_mm2):
    if df_vessels is None or df_vessels.empty: return None
    n = len(df_vessels)
    # Porosidade baseada em Pixels (Mask padrão 1024x768)
    porosidade = (df_vessels['Area_px'].sum() / (1024 * 768)) * 100
    return {
        "Arquivo": filename, "Nº Vasos": n, "Freq. (v/mm²)": n / img_area_mm2,
        "Média do Ø Maior (µm)": df_vessels['Major_Axis_um'].mean(),
        "Ø Maior Std": df_vessels['Major_Axis_um'].std(),
        "Média do Ø Menor (µm)": df_vessels['Minor_Axis_um'].mean(),
        "Ø Menor Std": df_vessels['Minor_Axis_um'].std(),
        "Área Média (µm²)": df_vessels['Area_um2'].mean(),
        "Área Std": df_vessels['Area_um2'].std(),
        "Porosidade (%)": porosidade
    }

def calcular_linhas_resumo(df_base, label_col_name="Quadrante"):
    cols_num = df_base.select_dtypes(include=[np.number]).columns
    row_sum = df_base[cols_num].sum(); row_sum[label_col_name] = "TOTAL"
    if "Freq. (v/mm²)" in row_sum: row_sum["Freq. (v/mm²)"] = None
    if "Porosidade (%)" in row_sum: row_sum["Porosidade (%)"] = None
    
    row_mean = df_base[cols_num].mean(); row_mean[label_col_name] = "MÉDIA"
    
    mapa_std = {"Ø Maior Std": "Média do Ø Maior", "Ø Menor Std": "Média do Ø Menor", "Área Std": "Área Média"}
    col_names = df_base.columns
    for col_std in col_names:
        for k_std, k_mean in mapa_std.items():
            if k_std in col_std:
                match = [c for c in col_names if k_mean in c]
                if match:
                    row_sum[col_std] = df_base[match[0]].std()
                    row_mean[col_std] = df_base[match[0]].std()
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
    
    # Filtra DataFrames: remove os que estão vazios ou que possuem apenas NAs
    dfs_to_concat = [
        d.dropna(axis=1, how='all') for d in [df, df_sum, df_mean] 
        if not d.empty and not d.isna().all().all()
    ]
    
    if dfs_to_concat:
        return pd.concat(dfs_to_concat, ignore_index=True).convert_dtypes()
    return df

# --- 5. SIDEBAR COMPLETO ---
with st.sidebar:
    st.header("Configurações")
    
    # --- SEÇÃO DE MODELOS ---
    st.subheader("Modelo de Inferência")
    model_dir = os.path.join(root_dir, "model")
    opcoes_onnx = listar_modelos_onnx(model_dir)
    
    if opcoes_onnx:
        model_name = st.selectbox("Selecione o modelo de IA", options=opcoes_onnx)
        onnx_path = os.path.join(model_dir, model_name)
    else:
        st.warning("Nenhum modelo de IA encontrado.")
        onnx_path = None
    
    st.divider()

    # --- CALIBRAÇÃO DE ESCALA ---
    st.subheader("Calibração de Escala")
    st.info("Informe a resolução (µm/px) da imagem original.")
    
    pixel_size_val = st.number_input(
        "Resolução (µm/px):", 
        value=1.0638, 
        format="%.4f",
        help="Quantidade de micrômetros por pixel na captura original do microscópio."
    )

    st.divider()

    # --- FILTROS DE ANATOMIA (QWA) ---
    st.subheader("Filtros de Segmentação")
    
    # Entrada amigável para o anatomista em micrometros quadrados
    min_area_um2 = st.number_input(
        "Área Mínima do Vaso (µm²):", 
        value=1000.0, 
        step=100.0,
        help="Vasos com área física menor que esta serão ignorados pelo processador."
    )

    # Cálculo do min_area_obj em pixels para o MaskPostProcessor.
    min_area_obj = int(round(min_area_um2 / (pixel_size_val ** 2)))
    
    # REMOVIDO: Equivalente técnico em pixels (conforme solicitado)

    THRESHOLD_FIXO = st.slider("Threshold de Confiança:", 0.1, 0.9, 0.5, 0.05)
    
    st.divider()
    
    st.subheader("Opções de Análise")
    ignorar_bordas = st.checkbox("Excluir vasos cortados?", value=False)
    save_masks = st.checkbox("Salvar Máscaras em Disco?", value=False)
    
    default_out = "host/data/output_results" if os.path.exists("/app/host") else "output_results"
    output_dir_name = st.text_input("Pasta Saída:", value=default_out)
    
# --- 6. APP PRINCIPAL ---
st.title("🔬 Relatório de Anatomia")

# Inicializa TODAS as chaves de estado necessárias no início
if 'results_raw' not in st.session_state: 
    st.session_state['results_raw'] = []
if 'uploaded_files_map' not in st.session_state: 
    st.session_state['uploaded_files_map'] = {}
if 'pixel_size_map' not in st.session_state: 
    st.session_state['pixel_size_map'] = {}
if 'processado' not in st.session_state: 
    st.session_state['processado'] = False

uploaded_files = st.file_uploader("Selecione as imagens:", accept_multiple_files=True)

# BOTÃO DE PROCESSAMENTO
if uploaded_files:
    if st.button("🚀 Gerar Relatório", type="primary"):
        # Limpeza e reinicialização para novo processamento
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            os.makedirs(TEMP_DIR)
            
        st.session_state['results_raw'] = []
        st.session_state['uploaded_files_map'] = {f.name: f for f in uploaded_files}
        st.session_state['processado'] = False # Reseta até terminar
        
        with st.spinner("Processando..."):
            adapter, erro = load_onnx_model(onnx_path)
            if not adapter: 
                st.error(erro)
                st.stop()
            
            bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                try:
                    img_pil = Image.open(file).convert("RGB")
                    orig_w, orig_h = img_pil.size
                    
                    # Cálculo do fator de escala e min_area dinâmico
                    fator_escala = calculate_area_scale_factor(orig_w, orig_h)
                    min_area_scaled = int(round((1/fator_escala) * min_area_obj))
                    
                    # Processador atualizado (Post-processing com regionprops)
                    post_proc = MaskPostProcessor(threshold=THRESHOLD_FIXO, min_area=min_area_scaled)
                    
                    mask_array = run_inference(adapter, img_pil, post_proc, min_area=min_area_scaled)
                    
                    temp_path = os.path.join(TEMP_DIR, f"temp_{file.name}.png")
                    Image.fromarray(mask_array).save(temp_path)
                    
                    df_img = calculate_qwa_metrics(temp_path, orig_w, orig_h, pixel_size_val)
                    
                    if df_img is not None and not df_img.empty:
                        df_img.insert(0, 'Arquivo', file.name)
                        df_img['Img_Area_mm2'] = ((orig_w * orig_h) * (pixel_size_val ** 2)) / 1_000_000.0
                        st.session_state['results_raw'].append(df_img)
                        st.session_state['pixel_size_map'][file.name] = pixel_size_val
                except Exception as e:
                    st.error(f"Erro em {file.name}: {e}")
                bar.progress((i+1)/len(uploaded_files))
        
        st.session_state['processado'] = True
        st.success("Processamento Concluído!")
        st.rerun() # Força a atualização para mostrar os resultados

# --- EXIBIÇÃO DOS RESULTADOS ---
if st.session_state['processado']:
    summary_list = []
    for df_raw in st.session_state['results_raw']:
        df_filtered = filtrar_vasos(df_raw, ignorar_bordas)
        if not df_filtered.empty:
            filename = df_raw['Arquivo'].iloc[0]
            area_mm2 = df_raw['Img_Area_mm2'].iloc[0]
            stats = calcular_resumo_imagem(df_filtered, filename, area_mm2)
            if stats: summary_list.append(stats)
    
    # VERIFICAÇÃO DE VASOS ENCONTRADOS
    if not summary_list:
        st.warning("⚠️ **Nenhum vaso foi detectado nas imagens processadas.**")
        st.info("""
            **Sugestões para ajuste:**
            1. **Revise a Imagem:** Verifique se a qualidade ou o contraste da captura original permite a identificação dos vasos.
            2. **Ajuste o Filtro:** A 'Área Mínima (µm²)' pode estar muito alta, excluindo todos os vasos detectados.
            3. **Troque o Modelo:** O modelo de IA selecionado pode não ser o mais adequado para este tipo específico de madeira.
        """)
    else:
        # TUDO O QUE DEPENDE DE DADOS FICA DENTRO DESTE ELSE
        df_summary = pd.DataFrame(summary_list)
        
        # --- RESUMO GLOBAL ---
        cols = ["Arquivo", "Nº Vasos", "Freq. (v/mm²)", "Média do Ø Maior (µm)", "Ø Maior Std", "Área Média (µm²)", "Porosidade (%)"]
        final_cols = [c for c in cols if c in df_summary.columns]
        df_base = df_summary[final_cols]
        
        df_sum, df_mean = calcular_linhas_resumo(df_base, label_col_name="Arquivo")
        
        # Filtra DataFrames para evitar o FutureWarning do Pandas 2.x
        dfs_to_concat = [
            d.dropna(axis=1, how='all') for d in [df_base, df_sum, df_mean] 
            if not d.empty and not d.isna().all().all()
        ]
        
        if dfs_to_concat:
            df_final = pd.concat(dfs_to_concat, ignore_index=True).convert_dtypes()
        else:
            df_final = df_base.copy()
        
        st.subheader("📋 Resumo Global")
        if ignorar_bordas:
            st.caption("⚠️ Exibindo apenas vasos inteiros. Vasos cortados pela borda foram excluídos.")
        else:
            st.caption("ℹ️ Exibindo todos os vasos (incluindo bordas).")

        height_global = 600 if len(df_final) > 25 else "content"

        st.dataframe(
            df_final.style.format(precision=2, na_rep="-").apply(
                lambda x: ['background-color: #e6e9ef; color: #000000' if x['Arquivo'] in ['TOTAL', 'MÉDIA'] else '' for i in x], axis=1),
            width="stretch", 
            hide_index=True,
            height=height_global
        )
        
        # --- DOWNLOADS ---
        final_out_dir = os.path.join(root_dir, output_dir_name)
        if not os.path.exists(final_out_dir): os.makedirs(final_out_dir)
        
        csv_sum = os.path.join(final_out_dir, "resumo_anatomico.csv")
        df_summary.to_csv(csv_sum, sep=';', encoding='utf-8-sig', index=False)
        
        df_all_raw = pd.concat(st.session_state['results_raw'], ignore_index=True)
        if "Imagem" in df_all_raw.columns:
            df_all_raw.drop(columns=["Imagem"], inplace=True)
            
        csv_raw = os.path.join(final_out_dir, "dados_brutos_vasos.csv")
        df_all_raw.to_csv(csv_raw, sep=';', encoding='utf-8-sig', index=False)

        c_dl1, c_spacer, c_dl2 = st.columns([2, 4, 2])
        with c_dl1:
            with open(csv_sum, "rb") as f: 
                st.download_button("📥 Baixar Resumo (CSV)", f, "resumo_anatomico.csv", width="stretch")
        with c_dl2:
            with open(csv_raw, "rb") as f: 
                st.download_button("📥 Dados Brutos (Completo)", f, "dados_brutos_vasos.csv", width="stretch")

        st.divider()

        # --- 2. VISUALIZADOR INTERATIVO ---
        st.header("🔍 Visualização & Detalhes")
        
        arquivos_disponiveis = df_summary['Arquivo'].tolist()
        selected_file = st.selectbox("Selecione uma imagem para detalhar:", options=arquivos_disponiveis)
        
        if selected_file:
            exibir_visualizacao = st.checkbox("📸 Exibir Imagem e Máscara (Segmentação)", value=False)
            
            df_raw_total = pd.concat(st.session_state['results_raw'], ignore_index=True)
            df_file_full = df_raw_total[df_raw_total['Arquivo'] == selected_file]
            df_file_filtered = filtrar_vasos(df_file_full, ignorar_bordas)
            
            if exibir_visualizacao:
                col_vis1, col_vis2 = st.columns(2)
                
                uploaded_file_ref = st.session_state['uploaded_files_map'].get(selected_file)
                pixel_size_used = st.session_state['pixel_size_map'].get(selected_file, pixel_size_val)
                temp_mask_path = os.path.join(TEMP_DIR, f"temp_{selected_file}.png")
                
                if uploaded_file_ref and os.path.exists(temp_mask_path):
                    img_original = Image.open(uploaded_file_ref).convert("RGB")
                    orig_w_ref, _ = img_original.size
                    
                    with col_vis1:
                        st.image(img_original, caption=f"Original: {selected_file}", width="stretch")
                    
                    mask_gray = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
                    mask_viz = desenhar_grid_quadrantes(
                        mask_gray, 
                        pixel_size_used, 
                        original_w=orig_w_ref, 
                        df_vessels=df_file_filtered
                    )
                    
                    with col_vis2:
                        st.image(mask_viz, caption="Segmentação + Quadrantes (1mm²)", width="stretch")

            st.subheader(f"📊 Estatísticas por Quadrante: {selected_file}")
            
            img_area_mm2_saved = df_file_full['Img_Area_mm2'].iloc[0]
            df_q = agrupar_por_quadrante(df_file_filtered, img_area_mm2_saved)
            
            st.dataframe(
                df_q.style.format(precision=2, na_rep="-").apply(
                    lambda x: ['background-color: #e6e9ef; color: #292933' if x['Quadrante'] in ['TOTAL', 'MÉDIA'] else '' for i in x], axis=1),
                width="stretch", 
                hide_index=True
            )

elif uploaded_files:
    st.info("Clique em 'Gerar Relatório' para iniciar.")
else:
    st.info("Faça upload das imagens para começar.")