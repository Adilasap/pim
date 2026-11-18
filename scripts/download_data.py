"""Download the raw datasets and verify their SHA-256 checksums.

1. Birbilis, Ghorbani & Zhao (2025) "A Database of Mechanical Properties of Steels",
   Mendeley Data, V1, doi:10.17632/jmwb9ddd43.1  (CC BY 4.0) - 3,234 records.
2. Citrine Informatics "steel_strength" via Matminer (figshare file 13354691) - 312 records.
"""
import _common  # noqa: F401
import hashlib
import urllib.request

from steel_pinn.config import BENCHMARK_JSON, PRIMARY_XLSX, RAW

FILES = [
    (PRIMARY_XLSX,
     "https://data.mendeley.com/public-files/datasets/jmwb9ddd43/files/"
     "cc944ab8-d3c8-448c-ac1d-172e8cc7e11d/file_downloaded",
     "eaa89971150cff2bd97d8a1fe20b21e908731fcf755b37155608b1200235b484"),
    (BENCHMARK_JSON, "https://ndownloader.figshare.com/files/13354691",
     "e36501d7057cd833223bb8ed9948668b5ac90fd585d29a749f45af51c1d7f6ad"),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for path, url, digest in FILES:
        if path.exists() and sha256(path) == digest:
            print(f"ok      {path.name}")
            continue
        print(f"fetch   {path.name}")
        urllib.request.urlretrieve(url, path)
        got = sha256(path)
        if got != digest:
            raise SystemExit(f"checksum mismatch for {path.name}: {got}")
        print(f"ok      {path.name}")


if __name__ == "__main__":
    main()
