#!/usr/bin/env python3
"""Build the predeclared PW-0318 one-row decode-authority bundle."""

try:
    from tools.build_pw0316_layer4_bundle import main
except ModuleNotFoundError:
    from build_pw0316_layer4_bundle import main


if __name__ == "__main__":
    raise SystemExit(main("PW-0318"))
