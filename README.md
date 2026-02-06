# QWA Automator: Segmentação de Vasos em Microscopia de Madeira

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/gchagas/QWA-Automator)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Container-2496ED)](https://www.docker.com/)

## 📄 Sobre o Projeto

Este projeto consiste no desenvolvimento de um método automático para **segmentação de vasos em imagens de microscopia de madeira de eucalipto**.

A ferramenta foi desenvolvida como parte da pesquisa de **Mestrado em Ciência da Computação (PPGCC)** na **Universidade Federal de São Paulo (UNIFESP)**. O objetivo é auxiliar na Anatomia Quantitativa da Madeira (QWA), automatizando a contagem e medição da área do lúmen dos vasos, métricas essenciais para análises de qualidade da madeira.

---

## 🚀 Funcionalidades

- **Upload de Imagens:** Suporte para imagens de microscopia (JPG, PNG).
- **Segmentação Automática:** Utiliza modelos de Deep Learning (YOLO/Ultralytics) para detectar vasos.
- **Cálculo de Métricas QWA:**
  - Contagem total de vasos.
  - Área média do lúmen.
  - Fração de área de vasos.
- **Visualização Interativa:** Interface amigável construída com Streamlit.
- **Exportação:** Download dos resultados (CSV) e das máscaras de segmentação.

---

## 🌐 Demonstração Online

O projeto está implantado e rodando publicamente no Hugging Face Spaces:

👉 **[Acessar QWA Automator Online](https://huggingface.co/spaces/gchagas/QWA-Automator)**
