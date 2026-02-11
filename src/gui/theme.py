import os
import matplotlib.colors as mcolors

class QWATheme:
    """
    Gerenciador de identidade visual do QWA-Automator.
    Extraído do logotipo oficial e expandido com cores complementares.
    """
    
    # --- Cores Principais (Do Logotipo) ---
    PRIMARY = "#0F2841"    # Azul Marinho Profundo (Texto principal e anel externo)
    SECONDARY = "#2D9CDB"  # Azul Ciano Vibrante (Anéis internos e destaque "AI")
    
    # --- Cores Complementares (Sugeridas) ---
    ACCENT = "#F2994A"     # Laranja Suave (Para botões de ação/CTA - Contraste direto com o azul)
    SUCCESS = "#27AE60"    # Verde Esmeralda (Para mensagens de sucesso/conclusão)
    WARNING = "#F2C94C"    # Amarelo Solar (Para alertas)
    ERROR = "#EB5757"      # Vermelho Coral (Para erros - menos agressivo que vermelho puro)
    
    # --- Neutros ---
    BACKGROUND = "#FFFFFF" # Branco Puro
    SURFACE = "#F8F9FA"    # Cinza muito claro (Para fundos de cards/áreas laterais)
    TEXT_MAIN = "#0F2841"  # O mesmo dark blue (Melhor que preto puro para leitura)
    TEXT_LIGHT = "#828282" # Cinza médio para legendas
    
    @classmethod
    def get_palette_hex(cls):
        """Retorna lista das cores principais para gráficos."""
        return [cls.PRIMARY, cls.SECONDARY, cls.ACCENT, cls.SUCCESS, cls.ERROR]

    @classmethod
    def get_cmap_rgb(cls):
        """Retorna paleta convertida para tuplas (0-1) compatível com Matplotlib."""
        hex_list = cls.get_palette_hex()
        return [mcolors.hex2color(h) for h in hex_list]

# --- Exemplo de Uso Rápido ---
if __name__ == "__main__":
    print(f"Cor Principal: {QWATheme.PRIMARY}")
    print(f"Cor de Ação: {QWATheme.ACCENT}")

class QWAAssets:
    """
    Gerenciador de caminhos de arquivos estáticos (Imagens, Logos, Ícones).
    Calcula os caminhos absolutos para funcionar no Docker e Local.
    """
    
    # Define a raiz do projeto baseada na localização deste arquivo
    # Ajuste o '..' dependendo de onde este arquivo .py estiver salvo
    # Supondo que esteja em src/gui/theme.py, subir 2 níveis chega na raiz.
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
    
    _ASSETS_DIR = os.path.join(_PROJECT_ROOT, "assets")

    # --- Definição dos Arquivos ---
    # Altere os nomes abaixo para corresponder exatamente aos seus arquivos
    _ICON_FILE = "logo_icon.png"
    _LOGO_HORIZ_FILE = "logo_horizontal.png"
    _LOGO_SQUARE_FILE = "logo_square.png"

    @classmethod
    def get_icon(cls):
        """Retorna caminho absoluto do ícone (favicon)."""
        return os.path.join(cls._ASSETS_DIR, cls._ICON_FILE)

    @classmethod
    def get_logo_horizontal(cls):
        """Retorna caminho absoluto da logo horizontal (Ideal para Sidebar)."""
        return os.path.join(cls._ASSETS_DIR, cls._LOGO_HORIZ_FILE)

    @classmethod
    def get_logo_square(cls):
        """Retorna caminho absoluto da logo quadrada (Ideal para Main/About)."""
        return os.path.join(cls._ASSETS_DIR, cls._LOGO_SQUARE_FILE)

    @classmethod
    def check_integrity(cls):
        """(Opcional) Verifica se os arquivos realmente existem."""
        missing = []
        for path in [cls.get_icon(), cls.get_logo_horizontal(), cls.get_logo_square()]:
            if not os.path.exists(path):
                missing.append(path)
        if missing:
            print(f"⚠️ AVISO: Assets não encontrados: {missing}")