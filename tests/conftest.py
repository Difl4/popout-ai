"""Configuração de caminhos para que os testes consigam importar o pacote src."""

import sys
from pathlib import Path

# Adiciona o diretório popout-ai/ ao sys.path para que 'src' seja importável como pacote
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
