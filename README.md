# Deepface Test

## 環境構築

```bash
apt install cmake

# Windowsの場合
choco install cmake
```

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install deepface
pip install tf-keras
```

## 実行

単一の顔が含まれたtarget画像と、その顔が含まれているかどうか調べたいcheck画像をそれぞれのフォルダに格納しておき、下記のように実行する。

```bash
python main.py path/to/target/img.jpg path/to/check/img.jpg
```

tmp_face_imgに対し、ファイル名に一致度を付したファイルが生成される。
値が小さいほど、一致度が大きい。
