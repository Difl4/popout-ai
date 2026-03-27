"""Entry point: python -m src [--cli]"""

import sys

def main() -> None:
    if "--cli" in sys.argv:
        from .interfaces.cli import run_cli_game
        run_cli_game()
    else:
        from .interfaces.gui import launch_gui
        launch_gui()

if __name__ == "__main__":
    main()
