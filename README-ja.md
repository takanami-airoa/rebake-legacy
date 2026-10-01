# Rebake: 🍞 ロボットデータをML対応の“焼きたて”フォーマットに

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Robots](https://img.shields.io/badge/Robots-HSR-blue)

> 🥖 生のロボットデータにメタ情報をまぶして、焼き直せば、学習にすぐ使えるLeRobotデータセットに。

## 概要

**現在、開発中です。APIやコマンドライン引数に変更が生じる可能性があります。**  
Rebake は、ロボティクスデータ収集と ML モデルトレーニング間の橋渡しを行い、**多様なロボット記録データを一貫した LeRobot 形式に統一**します。この拡張可能なツールは、異なるロボットプラットフォーム間のフォーマットの違いを解消し、研究者が模倣学習およびその他の ML アプリケーション用のデータセットをシームレスに結合・比較できるようにします。

## データセットへのリンク

> 🚧 **近日公開**

## サポートロボット

| ロボットプラットフォーム | ステータス  | データ形式 | 機能             | 設定  |
| ------------------------ | ----------- | ---------- | ---------------- | ----- |
| Toyota HSR               | ✅ 本番環境 | rosbag     | 完全変換、可視化 | `hsr` |

> **あなたのロボットをサポートして欲しいですか？** Issue を開く またはプラグインを貢献してください！

### 主要機能

- 🔄 **複数ロボット対応** (🔮 将来) - 様々なロボット用の拡張可能なプラグインアーキテクチャ
- 📊 **柔軟な処理モード** - 個別エピソードまたは統合データセット処理
- ☁️ **AWS Batch 統合** - 大規模データセット用のスケール処理
- 🎥 **自動可視化** - 動画および HTML 可視化の生成
- 📈 **データ管理ツール** - フィルタリング、マージ、データセット分析
- 🤖 **本番対応** - 現在 Toyota HSR をサポート、さらに多くのロボットが追加予定

## クイックスタート

### 前提条件

- Docker and docker-compose
- **Toyota HSR 記録データ** (meta.json メタデータ付き ROSbag 形式)
- 現在 HSR のみサポート - 他のロボットは近日公開

### 1 分セットアップ

```bash
# リポジトリをクローン
git clone <repository-url> rebake
cd rebake

# HSRデータセットディレクトリを設定
export HSR_DATASET_DIR=/path/to/your/rosbag/data

# コンテナをビルドして実行
cd docker
docker compose build hsr_data_converter
docker compose run hsr_data_converter
```

### 最初のデータセットを変換

```bash
# コンテナ内で、依存関係をインストール
GIT_LFS_SKIP_SMUDGE=1 uv sync

# rosbagをLeRobot形式に変換（HSR例）
uv run -m hsr_data_converter.rosbag2lerobot.main \
    --raw_dir /root/datasets \
    --out_dir ./output \
    --fps 10 \
    --robot_type hsr \
    --conversion_type individual
```

🎉 LeRobot データセットが`./output`に動画、メタデータ、構造化データと共に準備されます！

### データセットを確認

```bash
# 変換されたデータセットを可視化
uv run src/hsr_data_converter/visualize/lerobot_dataset.py \
    --repo-id your_dataset_name \
    --root ./output/{エピソードディレクトリ名} \
    --episode-index 0
```

## インストール

### Docker を使用（推奨）

提供された Docker 環境を使用するのが最も簡単です：

```bash
git clone <repository-url> rebake
cd rebake
git submodule update --init --recursive
cd docker
docker compose build hsr_data_converter
docker compose run hsr_data_converter
```

### ローカル開発セットアップ

開発用またはローカルインストールを好む場合：

```bash
# uvで依存関係をインストール
GIT_LFS_SKIP_SMUDGE=1 uv sync

# サブモジュールを初期化
git submodule update --init --recursive
```

## 使用方法

### 基本変換

HSR 記録データを LeRobot 形式に変換：

```bash
uv run -m hsr_data_converter.rosbag2lerobot.main \
    --raw_dir /path/to/rosbags \
    --out_dir /path/to/output \
    --fps 10 \
    --robot_type hsr \
    --conversion_type aggregate \
    --separate_per_primitive false
```

### 処理モード

- **`individual`**: 各 rosbag を個別のデータセットに変換
- **`aggregate`**: 複数の rosbag を単一のデータセットに結合

### データ管理

#### エピソードフィルタリング

条件に基づいて特定のエピソードを削除：

```bash
uv run src/hsr_data_converter/filter_episodes.py \
    --input_dataset_path ./input_dataset \
    --output_dataset_path ./filtered_dataset \
    --chunk_size 1000
```

#### データセット結合

複数のデータセットを結合：

```bash
uv run src/hsr_data_converter/merge_dataset.py \
    --sources ./dataset1 ./dataset2 \
    --output ./merged_dataset \
    --fps 10
```

#### データ可視化

データセット可視化を生成：

```bash
uv run src/hsr_data_converter/visualize/lerobot_dataset.py \
    --repo-id dataset_name \
    --root ./dataset_path \
    --episode-index 0
```

### データ形式要件

HSR 記録データは以下の構造に従う必要があります：

```
dataset_directory/
├── template-061707-25-04-30-09-01-51/
│   ├── data.bag
│   └── meta.json
├── template-061707-25-04-30-09-02-45/
│   ├── data.bag
│   └── meta.json
├── template-061707-25-04-30-09-03-36/
│   ├── data.bag
│   └── meta.json
├── template-061707-25-04-30-09-04-28/
│   ├── data.bag
│   └── meta.json
└── ...
```

> **注意**: 各エピソードディレクトリには`data.bag`ファイル（rosbag 記録）と`meta.json`ファイル（エピソードメタデータ）が HSR 固有のトピック構造で含まれています。

## 開発

### コード品質

```bash
# コードをフォーマット
make format

# リンティング実行 (ruff + mypy)
make lint

# テスト実行
make test

# カバレッジ付きテスト実行
make test-coverage
```

### 利用可能な Make コマンド

- `make format` - ruff でコードをフォーマット
- `make lint` - リンティングチェック実行 (ruff + mypy)
- `make test` - 全ユニットテスト実行
- `make test-coverage` - カバレッジレポート付きテスト実行
- `make ruff-check` - コードスタイルのみチェック
- `make ruff-fix` - コードスタイル問題を修正
- `make mypy` - 型チェック実行

### テスト

```bash
# 特定のテスト実行
uv run pytest tests/test_rosbag2lerobot.py -v

# カバレッジ付きで実行
make test-coverage
```

## トラブルシューティング

### よくある問題

1. **Docker ビルド失敗**: Docker と nVidia-docker が適切にインストールされていることを確認
2. **メモリエラー**: 大規模データセット用に Docker メモリ割り当てを増加
3. **権限エラー**: ファイル権限と Docker ボリュームマウントをチェック
4. **依存関係不足**: `git submodule update --init --recursive`を実行

### ヘルプ

- 🐛 GitHub Issuesで issue を報告

## コントリビューション

コントリビューションを歓迎します！バグ修正、機能追加、新しいロボットプラットフォームのサポートなど、あなたのサポートを感謝します。

**クイックスタート:**

1. リポジトリをフォーク
2. 機能ブランチを作成
3. 変更を加えてテストを追加
4. 品質チェックを実行: `make format && make lint && make test`
5. プルリクエストを開く

📋 **詳細な手順、開発セットアップ、ガイドラインについては、[コントリビューションガイド](CONTRIBUTING.md)をご覧ください。**

## ライセンス

このプロジェクトは Apache License 2.0 の下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルをご覧ください。

---

