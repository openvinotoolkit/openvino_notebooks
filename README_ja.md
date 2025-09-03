[English](README.md) | [簡体中文](README_cn.md) | 日本語

<h1>📚 OpenVINO™ ノートブック</h1>

[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/LICENSE)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/treon_precommit.yml?query=event%3Apush)
[![CI](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml/badge.svg?event=push)](https://github.com/openvinotoolkit/openvino_notebooks/actions/workflows/docker.yml?query=event%3Apush)

OpenVINO™ Toolkitを使って学習や実験を行うための、すぐに実行できるJupyterノートブックのコレクションです。これらのノートブックは、OpenVINOの基本を紹介し、最適化されたディープラーニング推論のためにAPIを活用する方法を開発者に教えます。

🚀 OpenVINO™ ノートブックの内容をナビゲートするためのインタラクティブなGitHubページアプリケーションをチェックしてください：
[OpenVINO™ ノートブック at GitHub Pages](https://openvinotoolkit.github.io/openvino_notebooks/)

[![notebooks-selector-preview](https://github.com/openvinotoolkit/openvino_notebooks/assets/41733560/a69efb78-1637-404c-b5ef-63974db2bf1b)](https://openvinotoolkit.github.io/openvino_notebooks/)

すべてのノートブックのリストは[インデックスファイル](./notebooks/README.md)で確認できます。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

## 目次

- [目次](#目次)
- [📝 インストールガイド](#-インストールガイド)
- [🚀 はじめに](#-はじめに)
- [⚙️ システム要件](#️-システム要件)
- [💻 ノートブックの実行](#-ノートブックの実行)
	- [単一のノートブックを起動する](#単一のノートブックを起動する)
	- [すべてのノートブックを起動する](#すべてのノートブックを起動する)
- [🧹 クリーンアップ](#-クリーンアップ)
- [⚠️ トラブルシューティング](#️-トラブルシューティング)
- [📊 テレメトリ](#-テレメトリ)
- [📚 追加リソース](#-追加リソース)
- [🧑‍💻 コントリビューター](#-コントリビューター)
- [❓ FAQ](#-faq)

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

<div id='-インストールガイド'/>

## 📝 インストールガイド

OpenVINOノートブックはPythonとGitを必要とします。始めるには、あなたのオペレーティングシステムまたは環境に適したガイドを選択してください：

| [Windows](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Windows) | [Ubuntu](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu) | [macOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/macOS) | [Red Hat](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [CentOS](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Red-Hat-and-CentOS) | [Azure ML](https://github.com/openvinotoolkit/openvino_notebooks/wiki/AzureML) | [Docker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Docker) | [Amazon SageMaker](https://github.com/openvinotoolkit/openvino_notebooks/wiki/SageMaker) |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()

<div id='-はじめに'/>

## 🚀 はじめに

この[ページ](https://openvinotoolkit.github.io/openvino_notebooks/)を使用してJupyterノートブックを探索し、あなたのニーズに関連するものを選択するか、すべて試してみてください。幸運を祈ります！

**注意：このリポジトリのmainブランチは新しいOpenVINO 2025.3リリースをサポートするように更新されました。** 新しいリリースバージョンにアップグレードするには、`openvino_env`仮想環境で`pip install --upgrade -r requirements.txt`を実行してください。初めてインストールする場合は、以下の[インストールガイド](#-インストールガイド)セクションを参照してください。以前のリリースバージョンのOpenVINOを使用する場合は、[2025.2ブランチ](https://github.com/openvinotoolkit/openvino_notebooks/tree/2025.2)をチェックアウトしてください。以前の長期サポート（LTS）バージョンのOpenVINOを使用する場合は、[2023.3ブランチ](https://github.com/openvinotoolkit/openvino_notebooks/tree/2023.3)をチェックアウトしてください。

助けが必要な場合は、GitHub [Discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions)を開始してください。  


問題が発生した場合は、[トラブルシューティングセクション](#-トラブルシューティング)、[FAQ](#-faq)を確認するか、GitHub [discussion](https://github.com/openvinotoolkit/openvino_notebooks/discussions)を開始してください。

![binderロゴ](https://mybinder.org/badge_logo.svg)と![colabロゴ](https://colab.research.google.com/assets/colab-badge.svg)ボタンが付いたノートブックは、何もインストールせずに実行できます。[Binder](https://mybinder.org/)と[Google Colab](https://colab.research.google.com/)は、リソースが限られた無料のオンラインサービスです。最高のパフォーマンスを得るには、[インストールガイド](#-インストールガイド)に従ってノートブックをローカルで実行してください。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-システム要件'></div>

## ⚙️ システム要件

これらのノートブックは、あなたのラップトップ、クラウドVM、またはDockerコンテナなど、ほぼどこでも実行できます。以下の表は、サポートされているオペレーティングシステムとPythonバージョンを示しています。

| サポートされているオペレーティングシステム                 | [Pythonバージョン (64-bit)](https://www.python.org/) |
| :--------------------------------------------------------- |:---------------------------------------------------|
| Ubuntu 20.04 LTS, 64-bit                                   | 3.9 - 3.12                                         |
| Ubuntu 22.04 LTS, 64-bit                                   | 3.9 - 3.12                                         |
| Red Hat Enterprise Linux 8, 64-bit                         | 3.9 - 3.12                                         |
| CentOS 7, 64-bit                                           | 3.9 - 3.12                                         |
| macOS 10.15.x バージョン以上                               | 3.9 - 3.12                                         |
| Windows 10, 64-bit Pro, Enterprise または Education エディション | 3.9 - 3.12                                         |
| Windows Server 2016 以上                                   | 3.9 - 3.12                                         |

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#)
<div id='-ノートブックの実行'></div>

## 💻 ノートブックの実行

### 単一のノートブックを起動する

単一のノートブック（例：Monodepthノートブック）のみを起動したい場合は、以下のコマンドを実行します（リポジトリのルートディレクトリから）：

```bash
jupyter lab notebooks/vision-monodepth/vision-monodepth.ipynb
```

### すべてのノートブックを起動する

インデックス`README.md`ファイルを開いて、ノートブックディレクトリとファイルの間を簡単にナビゲートできるようにJupyter Labを起動します。リポジトリのルートディレクトリから以下のコマンドを実行します：
	
```bash
jupyter lab notebooks/README.md
```

または、ブラウザでJupyter Labの左サイドバーを使用してファイルブラウザからノートブックを選択します。各チュートリアルは`notebooks`ディレクトリ内のサブディレクトリにあります。

<img src="https://user-images.githubusercontent.com/15709723/120527271-006fd200-c38f-11eb-9935-2d36d50bab9f.gif">

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-クリーンアップ'></div>

## 🧹 クリーンアップ

<div id='-shut-down-jupyter-kernel' markdown="1">

1. Jupyterカーネルをシャットダウンする

	Jupyterセッションを終了するには、`Ctrl-c`を押します。これにより、`このJupyterサーバーをシャットダウンしますか (y/[n])?`と表示されるので、`y`を入力して`Enter`を押します。
</div>	
	
<div id='-deactivate-virtual-environment' markdown="1">

2. 仮想環境を無効化する

	仮想環境を無効化するには、`openvino_env`をアクティブにしたターミナルウィンドウで単に`deactivate`を実行します。これにより、環境が無効化されます。

	環境を再アクティブ化するには、Linuxでは`source openvino_env/bin/activate`を、Windowsでは`openvino_env\Scripts\activate`を実行し、その後`jupyter lab`または`jupyter notebook`を入力して再度ノートブックを起動します。
</div>	

<div id='-delete-virtual-environment' markdown="1">

3. 仮想環境を削除する _(オプション)_

	仮想環境を削除するには、単に`openvino_env`ディレクトリを削除します：
</div>

<div id='-on-linux-and-macos' markdown="1">

  - LinuxおよびmacOSの場合：

	```bash
	rm -rf openvino_env
	```
</div>

<div id='-on-windows' markdown="1">

  - Windowsの場合：

	```bash
	rmdir /s openvino_env
	```
</div>

<div id='-remove-openvino-env-kernel' markdown="1">

  - Jupyterから`openvino_env`カーネルを削除する

	```bash
	jupyter kernelspec remove openvino_env
	```
</div>

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-トラブルシューティング'></div>

## ⚠️ トラブルシューティング

これらのヒントで問題が解決しない場合は、[ディスカッショントピック](https://github.com/openvinotoolkit/openvino_notebooks/discussions)を開くか、[issue](https://github.com/openvinotoolkit/openvino_notebooks/issues)を作成してください！

- 一部の一般的なインストール問題を確認するには、`python check_install.py`を実行します。このスクリプトはopenvino_notebooksディレクトリにあります。
  `openvino_env`仮想環境をアクティブにした後に実行してください。
- `ImportError`が発生した場合は、Jupyterカーネルをインストールしたかどうかを再確認してください。必要に応じて、Jupyter LabまたはJupyter Notebookの_Kernel->Change Kernel_メニューから`openvino_env`カーネルを選択してください。
- OpenVINOがグローバルにインストールされている場合、`setupvars.bat`または`setupvars.sh`がソースされたターミナルでインストールコマンドを実行しないでください。
- Windowsインストールの場合、_Command Prompt (`cmd.exe`)_を使用することをお勧めします。_PowerShell_ではありません。
- `ImportError: cannot import name 'collect_telemetry' from 'notebook_utils'`エラーが発生した場合は、ノートブックディレクトリに最新の`notebook_utils.py`ファイルがダウンロードされていることを確認してください。古い`notebook_utils.py`ファイルを削除してノートブックを再実行してみてください。新しいユーティリティファイルがダウンロードされます。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-テレメトリ'></div>

## 📊 テレメトリ

`collect_telemetry()`関数を含むノートブックセルを実行すると、テレメトリデータが収集され、エクスペリエンスの向上に役立ちます。
このデータはセルが実行されたことを示すだけで、個人を特定できる情報（PII）は含まれません。

デフォルトでは、匿名のテレメトリデータが収集され、ノートブックの実行に限定されます。
このテレメトリは、Intelのソフトウェア、ハードウェア、ウェブサイト、または製品には拡張されません。

テレメトリを無効にしたい場合は、ノートブック内のデータ収集を担当する特定の行をコメントアウトすることで、いつでも無効にできます：
```python
# collect_telemetry(...)
```
また、`SCARF_NO_ANALYTICS`または`DO_NOT_TRACK`環境変数を`1`に設定することで、テレメトリ収集を無効にすることもできます：
```bash
export SCARF_NO_ANALYTICS=1
# または
export DO_NOT_TRACK=1
```

テレメトリの目的でScarfが使用されます。データの収集と処理方法については、[Scarfのドキュメント](https://docs.scarf.sh/)を参照してください。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-追加リソース'></div>

## 📚 追加リソース
* [OpenVINOブログ](https://blog.openvino.ai/) - OpenVINOのベストプラクティス、興味深いユースケース、チュートリアルを集めた技術記事���コレクション。
* [Awesome OpenVINO](https://github.com/openvinotoolkit/awesome-openvino) - OpenVINOベースのAIプロジェクトのキュレーションリスト。
* [OpenVINO GenAIサンプル](https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples) - OpenVINO GenAI APIサンプルのコレクション。
* [Edge AIリファレンスキット](https://github.com/openvinotoolkit/openvino_build_deploy) - 小売、医療、製造などのさまざまな業界での生産グレードのAIアプリケーションの開発と展開を加速するための事前構築されたコンポーネントとコードサンプル。
* [Open Model Zooデモ](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md) - 特定のディープラーニング推論シナリオを実装するためのテンプレートを提供するコンソールアプリケーション。これらのアプリケーションは、モデル推論のためのデータの前処理と後処理の方法を示し、処理パイプラインを整理します。
* [oneAPIサンプル](https://github.com/oneapi-src/oneAPI-samples)リポジトリは、マルチアーキテクチャ環境でのoneAPIとそのツールキット（oneDNNなど）が提供するパフォーマンスと生産性を示しています。OpenVINO™ツールキットは、oneAPIを使用してディスクリートGPUを活用し、マルチアーキテクチャプログラミングのためのオープンプログラミングモデルを提供します。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)](#-contributors)
<div id='-コントリビューター'></div>

## 🧑‍💻 コントリビューター

<a href="https://github.com/openvinotoolkit/openvino_notebooks/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openvinotoolkit/openvino_notebooks" />
</a>

[`contrib.rocks`](https://contrib.rocks)で作成。

[![-----------------------------------------------------](https://user-images.githubusercontent.com/10940214/155750931-fc094349-b6ec-4e1f-9f9a-113e67941119.jpg)]()
<div id='-faq'></div>

## ❓ FAQ

* [OpenVINOはどのデバイスをサポートしていますか？](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes.html)
* [OpenVINOがサポートする最初のCPU世代は何ですか？](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/system-requirements.html)
* [OpenVINOを使用して現実世界のソリューションを展開する成功事例はありますか？](https://www.intel.com/content/www/us/en/internet-of-things/ai-in-production/success-stories.html)

---

\* 他の名前やブランドは他者の財産として主張される場合があります。

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=README_ja.md" />

Human Rights Information: “Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See Intel’s Global Human Rights Principles at https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf. Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
