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
pip install deepface tf-keras ultralytics
```

## 実行

- 単一の顔が含まれたtarget画像を複数含むディレクトリ
- その顔が含まれているかどうか調べたいcheck画像

```bash
python main.py path/to/target/dir/ path/to/check/img.jpg
```

tmp_face_imgに対し、ファイル名に一致度を付したファイルが生成される。
値が小さいほど、一致度が大きい。

# 類似度について

- 0.4 未満: 同一人物の可能性が非常に高い
- 0.4 ~ 0.6: グレーゾーン（慎重な判断が必要）
- 0.6 以上: 異なる人物の可能性が高い

ここでは、複数の画像から、一致度が最大のものを抽出するようにしている。
ざっと、0.5未満であれば一致しているとしても差し支えなさそうには見える。
