"""Split dataset baut vs mur menjadi train/val."""
from pathlib import Path
import csv, random, shutil

RAW = Path('dataset_raw'); OUT = Path('dataset'); CLASSES = ['baut','mur']; VAL_RATIO = 0.2; SEED = 42

def main():
    random.seed(SEED)
    groups = {}
    meta = RAW / 'metadata.csv'
    if meta.exists():
        with meta.open(newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
        if rows and 'session_id' in rows[0]:
            for row in rows:
                cls, name, session = row.get('kelas',''), row.get('nama_file',''), row.get('session_id','')
                if cls in CLASSES and session: groups.setdefault((cls, session), []).append(RAW / cls / name)
    train, val = {c: [] for c in CLASSES}, {c: [] for c in CLASSES}
    if groups:
        for cls in CLASSES:
            gs = [(k, v) for k, v in groups.items() if k[0] == cls]; random.shuffle(gs)
            n = max(1, round(len(gs) * VAL_RATIO))
            for _, files in gs[:n]: val[cls].extend(p for p in files if p.exists())
            for _, files in gs[n:]: train[cls].extend(p for p in files if p.exists())
    else:
        for cls in CLASSES:
            files = [p for p in (RAW / cls).glob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
            random.shuffle(files); n = max(1, round(len(files) * VAL_RATIO))
            val[cls], train[cls] = files[:n], files[n:]
    for split in ['train','val']:
        for cls in CLASSES:
            dest = OUT / split / cls; dest.mkdir(parents=True, exist_ok=True)
            for old in dest.iterdir():
                if old.is_file(): old.unlink()
    for split, data in [('train', train), ('val', val)]:
        for cls, files in data.items():
            dest = OUT / split / cls
            for src in files: shutil.copy2(src, dest / src.name)
    for cls in CLASSES: print(f'{cls}: train={len(train[cls])}, val={len(val[cls])}')

if __name__ == '__main__': main()