import kagglehub
import zipfile
import shutil
from pathlib import Path

try:
    dst = Path("data/students-performance")
    dst.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download("glipko/additional-data-predict-students-performance")
    print("Downloaded:", path)
    p = Path(path)

    if p.is_dir():
        print("Dataset path is a directory; copying directory tree to", dst)
        # copytree with dirs_exist_ok to merge into existing dst
        shutil.copytree(p, dst, dirs_exist_ok=True)
    elif p.suffix in ('.zip', '.gz', '.tar') or str(p).endswith('.tar.gz'):
        print("Extracting archive to", dst)
        if str(p).endswith('.zip'):
            with zipfile.ZipFile(p, 'r') as z:
                z.extractall(dst)
        else:
            shutil.unpack_archive(str(p), str(dst))
    else:
        # If it's a single file, just copy
        if p.is_file():
            shutil.copy(p, dst / p.name)
        else:
            print("Unhandled path type:", p)

    print("Contents of", dst)
    for f in dst.rglob('*'):
        print(f.relative_to(dst))

except Exception as e:
    print("Error:", e)