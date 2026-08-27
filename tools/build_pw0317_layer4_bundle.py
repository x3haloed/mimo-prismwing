#!/usr/bin/env python3
"""Build the predeclared PW-0317 layer-4 three-K4/five-source bundle."""

try:
    from tools.build_pw0316_layer4_bundle import main
except ModuleNotFoundError:
    from build_pw0316_layer4_bundle import main


if __name__ == "__main__":
    raise SystemExit(main("PW-0317"))
