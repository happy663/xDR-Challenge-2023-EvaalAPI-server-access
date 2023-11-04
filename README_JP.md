# xDR-Challenge-2023-EvaalAPI-server-access

xDR Challenge 2023 コンペティション (IPIN 2023 コンペティション トラック 5)において，テストトライアルおよびスコアリングトライアルのデータを Evaal API サーバからダウンロードする方法，ならびに軌跡推定結果を Evaal API サーバへ提出する方法の詳細については，こちらのサイト（https://evaal.aaloa.org/2023/software-for-offsite-tracks）で説明されています．
以下は今回配布するサンプルプログラムの説明になりますが，今回のサンプルでは，Evaal API サーバを用いたトライアルデータダウンロードおよび軌跡推定結果ファイル提出の手順を実際に試し，これらに伴うサーバの動作を確認することができます．

「トライアルネーム」は，データダウンロードおよび軌跡推定結果ファイル提出を行う際に必要になりますので，info@evaal.aaloa.org宛にトライアルネームの配布を依頼して，事前に入手しておいてください．

## 必要条件

```
requests==2.29.0
```

## ファイルの概要

| **ファイル名**             | **概要**                                              |
| -------------------------- | ----------------------------------------------------- |
| do_downloading_trials.py   | トライアルデータダウンロード用スクリプト              |
| do_submitting_estimates.py | 推定軌跡ファイル提出用スクリプト                      |
| requirements.txt           | Python の必要ライブラリのバージョンを記述したファイル |

## 使用方法

トライアルネームを入手後に以下のステップを実行してください．

### Step.1 インストール

```
git clone --recursive https://github.com/PDR-benchmark-standardization-committee/xDR-Challenge-2023-download-submission
cd xDR-Challenge-2023-download-submission
pip install -r requirements.txt
```

### Step.2 実行スクリプトとフォルダの配置

一例として次のようにフォルダを配置してください．
フォルダの相対パスは，以降で説明するコマンドで正しく入力される必要があります．

```
xDR-Challenge-2023-evaluation/
├ dataset/
|   ├ traj/     (to be used for submission)
|   └ trials/   (to be used for download)
|
├ do_downloading_trials.py
├ do_submitting_estimates.py
├ requirements.txt
└ README.md
```

### Step.3 トライアルデータのダウンロード

1 つのトライアルネームは，対応する 1 組のトライアルデータと推定軌跡ファイルのダウンロードあるいは提出に使うことができます．
トライアルデータをダウンロードする場合は，次のスクリプトを実行してください．

```
python do_downloading_trials.py [trial_name] [server_url] ./dataset/trials/[give_the_name_as_you_like].txt
```

正常動作の場合には，レスポンスコード 200 を受け取るとともに，トライアルデータが指定したフォルダに保存されます． データフォーマットについては，公式ページの README
(https://unit.aist.go.jp/harc/xDR-Challenge-2023/data/README_jp.md)
の"データ形式"節を参照してください．

### Step.4 軌跡推定の実行

自身のチームで軌跡推定のために用意したプログラムを実行させて，推定軌跡ファイルを作成してください．
軌跡ファイルの中身は，以下の順にデータを並べた上でカンマ区切りの構成としてください．

```
Timestamp(s),x(m),y(m),floor(FLU01/FLU02/FLD01)
```

公式ページ提供のデータセットの最新版 (https://unit.aist.go.jp/harc/xDR-Challenge-2023/data/xdrchallenge2023_dev_0712.zip) には，このフォーマットに則った推定軌跡ファイルを生成する想定の軌跡推定デモスクリプト（02_output_example.ipynb）がありますので参考にしてください．ただし以下の注意事項については，自身のチーム作成のプログラム中で満たすようにしてください．

[注意事項]

- 軌跡ファイル内にヘッダーは含みません．
- ファイル全体をタイムスタンプ列で昇順にソートしてください．提出失敗の原因になるため，タイムスタンプの時刻が遡ることのないようにしてください．
- タイムスタンプは固定小数点表記および非負値としてください．指数表記あるいはマイナスの値のタイムスタンプは提出失敗の原因になります．

### Step.5 推定軌跡ファイルの提出

作成した推定軌跡ファイルを[dataset]/[traj]/フォルダー内に配置（コピー）してください．
その後，次ののスクリプトを実行してください．

```
python do_submitting_estimates.py [trial_name] [server_url] ./dataset/traj/[your_file_name_of_estimated trajectory].txt
```

正常動作の場合には，レスポンスコード 201 を受け取るとともに，提出した推定軌跡ファイルがサーバにおいて受理されます．
