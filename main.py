"""Atalho para rodar pelo PyCharm (Shift+F10).

Sem argumentos, assiste o agente guloso jogar a fase padrão.
Para os outros comandos, veja: python -m pvz --help
"""
import sys

from pvz.__main__ import main

if __name__ == "__main__":
    main(sys.argv[1:] or ["assistir"])
