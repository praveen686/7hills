import argparse
import asyncio
from .server import main

parser = argparse.ArgumentParser(description="QuantLaxmi Dashboard")
parser.add_argument("--port", type=int, default=8080)
args = parser.parse_args()
asyncio.run(main(args.port))
