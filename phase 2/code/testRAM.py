from pathlib import Path

import pandas as pd
from console import print_status

data_dir = Path(__file__).resolve().parent / "raw_datas"

files = [
    "hvfhv_2024_oct-dec_2p5pct.csv",
    "hvfhv_2025_2p5pct.csv",
    "hvfhv_2026_jan-sep_2p5pct.csv",
]

df = pd.concat(
    [pd.read_csv(data_dir / filename) for filename in files],
    ignore_index=True,
)

print_status(f"بارگذاری کامل شد. {len(df):,} ردیف در RAM است.")

while input("برای خروج 1 را وارد کن و Enter بزن. ").strip() != "1":
    pass
