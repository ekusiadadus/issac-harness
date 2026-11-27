# 公開事例から採用した設計

2026-11-25（日本時間）に原典を確認。公開事例の事実と、プロジェクト共通の制作・検証手順、サンプルで試した内容を対応付けた。

| 出典 | 公開されている使い方 | このハーネスへの反映 |
|---|---|---|
| Amazon Robotics と NVIDIA | 倉庫の設計・運用を仮想空間で検討し、Proteus の開発に Isaac Sim のセンサーエミュレーションを活用する取り組み。Amazon の 2026 年発表では、協業対象に合成データと real-to-sim 検証を挙げている。[NVIDIA の Proteus 事例](https://blogs.nvidia.com/blog/gpu-aws-omniverse-isaac-sim-robots/) · [Amazon の協業発表](https://press.aboutamazon.com/aws/2026/8/aws-and-nvidia-to-deliver-2-million-additional-gpus-and-next-generation-infrastructure-for-agentic-and-physical-ai) | 設備にどのセンサーが何を見つけるかを定義。実装例はオリジナルの作業台と 130 × 59 mm の EAN-13 ラベル。視点を固定し、照明変更後も保存画像の指定領域から番号を読み取る |
| Isaac Lab-Arena | Scene、embodiment、task を合成し、条件を制御して評価する。参照時の main は v0.3.1 を基に Lab 3.0 へ対応中の開発版。[2026-11-24 のソース](https://github.com/isaac-sim/IsaacLab-Arena/blob/b7cb2585c06a01bb25e26f26eecb2a662c4697e5/README.md) | 環境を独立した USD とし、カメラ、タスクの対象、合格条件、比較ケースを外から指定。レンダラーを変えても同じ比較を再実行できる |
| RoboCasa365 | 評価用の `target` では事前学習と異なる対象物・シーンを使い、評価条件を公開する。[2026-11-18 の評価手順](https://github.com/robocasa/robocasa/blob/7bfc60c9d26c02506d4265cbe0e86b35e6e25d42/docs/benchmarking/benchmarking_overview.md) | ケース名・入力ハッシュ・視点別結果を保存。追加の照明や対象を増やすときは評価用条件を事前に固定する運用を制作ガイドに採用 |
| USD Working Group | アセットの公開入口、component と assembly、参照と再利用を整理している。[コミュニティのアセット構成ガイド](https://github.com/usd-wg/assets/blob/091bb5c6f8eb665ad4cf9d6c10025cecc07eda88/docs/asset-structure-guidelines.md) | 軽量な環境入口とバイナリの重い形状を分け、用途別の撮影レイヤーを合成。プロジェクトを別ディレクトリへ移して検証 |
| NVIDIA USD Validation | v1.23.0 が 2026-10-27 に公開。OpenUSD の構造、メッシュ、シェーダー等を独立した CLI で検査できる。[公式リリース](https://github.com/NVIDIA-Omniverse/usd-validation-nvidia/releases/tag/v1.23.0) | `check --nvidia` が既定ルールで各ケースを検査。制作中に見つかった extent、UV の重複、アスキー形式の大配列をビルダー側で修正 |
| NVIDIA SimReady | 実寸の形状、材質ごとの PBR 入力、UV 密度と素材の再利用。最近の制作事例では、ライブラリ資産を配置し、レンダー画像で調整している。[モデリング](https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/modeling-best-practices.html) · [材質とテクスチャ](https://docs.omniverse.nvidia.com/simready/latest/simready-asset-creation/material-best-practices.html) · [2026 年 11 月の制作事例](https://blogs.nvidia.com/blog/developers-simulation-frontier-ai-agents/) | 資産の由来・実寸・物性を記録し、オフラインで解決する USD にまとめる。全景・接写を撮り、視認性と材質を両方確認する |

## 検証対象の実行環境

- 作業対象を通常距離と接写の両方から撮影し、撮影結果と支持面の証拠を残す構成を採用した。支持面 4 組を合成済みの衝突形状から検査し、実行時は箱と端末の落下・静定を PhysX のテンソルで測る。
- GPU 実行対象は Isaac Sim 6.2.0、Isaac Lab v3.0.0、Python 3.12。Lab v3.0.0 の公式対応に合わせた。[Isaac Lab のリリース](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0) · [Isaac Sim 6.2 の撮影例](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_getting_started.html)

梱包作業台の寸法、照明、カメラ、バーコードの採用値はこのサンプルの設計値。Amazon の倉庫に関する固有寸法・照明設定・検出性能は参照値に使っていない。

## 製造検査データの事例と参照した実装

1 機種あたり 5 万枚を超える CAD 由来の合成画像を公開しているのは Amazon Devices & Services の製造検査事例。Amazon Robotics の Proteus は倉庫ロボットのセンサーと運用シミュレーションの別事例。前者の多様化と、後者のタスクからのセンサー設計を参考にした。[Amazon Devices の製造検査](https://blogs.nvidia.com/blog/amazon-zero-touch-manufacturing/) · [Amazon Robotics の Proteus](https://blogs.nvidia.com/blog/gpu-aws-omniverse-isaac-sim-robots/)

Isaac Sim 6.2 の公式スキルは黒画像・照明・実行ログの確認と、同じカメラの基準画像との差分可視化を提供する。`accept` はここに実写と原寸画像を観察した所見、画像のハッシュ、タスク固有のバーコード照合を加える。`compare` の対象は同じシミュレータで別々に撮影した画像。生成時に選んだ姿勢、材質、マスクと矩形の対応は個別フレームに記録する。[Isaac Sim Validator](https://github.com/isaac-sim/IsaacSim/tree/v6.2.0/skills/isaac-sim-validator) · [Validation Difference GIFs](https://github.com/isaac-sim/IsaacSim/tree/v6.2.0/skills/validation-diff-gifs) · [SimReady Foundation](https://github.com/NVIDIA/simready-foundation)

Amazon Robotics の Virtual Systems チームは、段ボールや紙を下地に、シール・ロゴ・損傷を順に重ねて包装の多様性を作る工程を説明している。このサンプルは印刷面、テープ、擦れ、切断面を独立して表現する。SceneSmith の著者らが公開したマルチエージェントによるシーン批評、物理的な支持・安定性の検査を参考に、実写・レンダーの画像レビューと PhysX の静定観測を個別に記録する。[Amazon Robotics チームへの Adobe インタビュー](https://blog.adobe.com/en/publish/2023/11/30/amazon-robotics-combines-power-nvidia-omniverse-adobe-substance-3d-simulate-warehouse-operations) · [SceneSmith の公式実装](https://github.com/nepfaff/scenesmith)

## 任意の環境への展開と検証

Isaac Sim 6.2 の Scene-Based Synthetic Dataset Generation は、ステージ・照明・対象物の組合せとカメラを変更して同じシーンから教師を生成する。ここでは別の USD にも適用できるよう対象 prim と操作する属性を設定に移し、Replicator が返すフレーム単位の ID を永続的なクラス・個体 ID に対応付ける。CPU の検査では、構成を変えた後の資産、フレーム内容、実際のレンダーに切り替わった後のマスクを再検査する。[NVIDIA のシーンベース生成例](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html) · [NVIDIA の Replicator annotator API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html)

BOP YCB-Video は物体のスキャン、カメラ行列、物体姿勢、RGB-D、可視マスクを提供する。校正デモの実写と物体モデルは BOP 版の同一リビジョンに固定した。検出評価は別系列の実写を使う。TUM Mono は実測の魚眼レンズ、カメラ応答、周辺減光を扱う独立したセンサープロファイルの例に用いた。[BOP YCB-Video](https://huggingface.co/datasets/bop-benchmark/ycbv) · [YCB-Video 原著者のページ](https://rse-lab.cs.washington.edu/projects/posecnn/) · [TUM Mono の測定と配布資料](https://cvg.cit.tum.de/data/datasets/mono-dataset)

画面の空白や画像サイズを検査し、レンダー時の個体 prim と PhysX の実測トレース（接触・静定）を別々の証拠として保存する。

Amazon Robotics の制作担当者は、段ボールの基材にシールやロゴを重ね、最後に損傷を加え、Omniverse から Python で形状と材質の両方を変えていると説明している。本ハーネスでは、同じクラスの形状バリアント、実寸の傷と登録された PBR テクスチャ、個体の一面だけに貼るデカールを独立して設定できる。現実のセンサー・照明・検出性能への適合は該当する撮影系列で測る。「機種ごとに 5 万枚超」は Amazon Devices & Services が製造検査に使った値で、公開された Amazon Robotics の認定基準ではない。[Amazon Robotics の制作担当者への取材](https://blog.adobe.com/en/publish/2023/11/30/amazon-robotics-combines-power-nvidia-omniverse-adobe-substance-3d-simulate-warehouse-operations) · [Amazon Devices & Services の製造検査](https://blogs.nvidia.com/blog/amazon-zero-touch-manufacturing/)
