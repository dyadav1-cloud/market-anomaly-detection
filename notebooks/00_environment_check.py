import torch
import numpy as np
import pandas as pd
import yfinance as yf

print("="*50)
print("ENVIRONMENT CHECK")
print("="*50)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"CUDA version:    {torch.version.cuda}")

print(f"\nPandas version: {pd.__version__}")
print(f"NumPy version:  {np.__version__}")

# Try to pull a tiny bit of data
print("\nPulling 5 days of SPY data...")
spy = yf.download("SPY", period="5d", progress=False)
print(spy)

print("\n Setup complete!")
