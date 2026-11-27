# 接触試験（Isaac Lab 3.0）

`runtime/contact_trials.py` は既存のトレーと箱の USD を孤立したステージに読み込み、PhysX tensor view から全ステップの位置、姿勢、並進・角速度を記録する。各試験は別プロセスで起動する。把持と開いた指の対照試験は、現在の PhysX GPU rigid-body view で運動学的ターゲット設定が未実装だったため、`--device` を明示しない場合は CPU 物理を選ぶ。明示した `--device` はそのまま使用し、GPU を選んだ場合は同じ未実装エラーになり得る。[Omni Physics tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html) も `set_kinematic_targets` はバックエンドが対応しない場合に例外を出すと記載する。横衝突と滑走は launcher の既定デバイスを使用する。`--visualizer none` を使うヘッドレスの物理試験で、画像品質の検査には使わない。

```sh
export HARNESS_RUNTIME=lab ACCEPT_EULA=Y
scripts/isaac_python.sh "$PWD/runtime/contact_trials.py" --trial rim \
  --asset "$PWD/examples/inspection/assets/tray/tray.usdc" \
  --report "$PWD/out/contact-rim.json" --visualizer none
scripts/isaac_python.sh "$PWD/runtime/contact_trials.py" --trial slide \
  --asset "$PWD/examples/inspection/assets/handset/handset.usdc" \
  --report "$PWD/out/contact-slide.json" --speed 0.6 --visualizer none
scripts/isaac_python.sh "$PWD/runtime/contact_trials.py" --trial grasp \
  --asset "$PWD/examples/inspection/assets/carton/carton.usdc" \
  --report "$PWD/out/contact-grasp.json" --visualizer none
scripts/isaac_python.sh "$PWD/runtime/contact_trials.py" --trial open-control \
  --asset "$PWD/examples/inspection/assets/carton/carton.usdc" \
  --report "$PWD/out/contact-open-control.json" --visualizer none
```

`rim` は、5 個のコライダーを持つ実物形状のトレーを固定治具にし、球を x 方向へ 1 m/s で入射する。縁の x 座標付近への到達、逆向きの x 速度、縁の貫通上限を検査する。`effective_rebound_ratio` は球とトレーの接触を含めたシミュレーション上の有効応答であり、実物の復元係数ではない。

`slide` は実際に制作した端末を広い板に置いて横速度を与え、減速する PhysX 軌跡に直線を当てはめる。最初の 5 ステップを除き、x 方向速度が 0.10 m/s を超え、x < 0.35 m の区間だけで `μ_eff = −(dv_x/dt)/9.81` を求める。8 点以上、`0.02 < μ_eff < 1.5`、速度の RMS 残差 0.15 m/s 未満を受入条件とする。回転・材質結合・solver を含むシミュレーション有効値である。

`grasp` は箱を支持面に置き、独立した運動学的な 2 本の指を閉じる。0.3 秒後から 1.2 秒かけて 180 mm 上昇し、保持後に開放する。箱に親子付け、固定ジョイント、アタッチ、位置の直接設定を行わない。保持区間の箱の高さと変動、開放後の落下を判定する。`open-control` は指を開いたまま同じ軌跡を動かし、箱が浮かないことを判定する。指の摩擦設定は引数 `--finger-friction` で変えられるが、実物との一致を示す値ではない。

支持面と指の治具は、指定寸法の頂点を持つ凸メッシュとして作り、剛体 prim に非一様な scale を付けない。把持試験の支持面は静的コライダーで、指だけが運動学的剛体である。CPU tensor view が reset 直後に指を原点へ置き、速度を与える場合があるため、計測開始前に両指の位置と運動学的ターゲットを同じ開位置に設定し、その場で 2 ステップ進めて速度を静定させる。箱を支持面に置き、動的剛体の 6 軸速度をゼロにして計測を始める。読み戻した位置と速度が許容範囲外なら `startup_failed` レポートを保存して停止する。通常レポートには reset 直後と補正後の状態を、それぞれ `post_reset_*` と `initial_*` に保存する。既定の箱では左右の衝突壁の外端は x = ±0.1625 m、開いた指の内面は x = ±0.179 m で、左右それぞれに 16.5 mm の隙間がある。フラップは y = ±0.235 m まで張り出すが、元アセットでは CollisionAPI を持たず、接触を作らない。別のアセットではこれらの寸法と衝突設定を再確認する。

出力 JSON は `status`、入力、バージョン、計算時間、PhysX ステップ数、全軌跡を持つ。判定不合格または runtime 例外では終了コードが非ゼロになる。実物の摩擦・復元係数や把持の成功率を主張するには、別途実測と複数条件の比較が必要である。

実物測定の時系列 CSV はこのリポジトリに存在せず、本試験では読み込んでいない。実物との係数一致は未評価である。

## 別のアセットを使う場合

各 USD は参照可能な `defaultPrim` と、その prim に適用された `RigidBodyAPI` を必要とする。ファイル名や `defaultPrim` の名前、総コライダー数は固定しない。`rim` は相対パス `--rim-collider`、`slide` は `--slide-collider`、把持は `--grasp-left-collider` と `--grasp-right-collider` を指定する。これらが合成後の USD 上で `CollisionAPI` を持つことを起動時に確認する。既定値はこのリポジトリのトレー、端末、箱に合わせた `RimRight`、`Body`、`WallLeft`、`WallRight`。

治具は対象の実寸に合わせて CLI で調整する。トレーへの入射では `--rim-probe-x`、`--rim-probe-z`、`--rim-probe-radius`、`--rim-contact-zone-x`、`--rim-precontact-x`、`--rim-max-x` を使う。把持では `--finger-open-x`、`--finger-closed-x`、`--finger-center-z`、`--finger-size-x/y/z`、`--grasp-lift-m`、`--support-top-z` を使う。各値はワールド座標のメートル単位。形状や質量が異なると既定の合否閾値が適切とは限らないため、判定の前に実寸、支持面、接触区間、ステップ間隔を確認する。入力値はレポートの `geometry_inputs` に保存する。必要な相対コライダーのパスは `required_colliders` に保存する。

例として別の箱の半幅が 0.10 m なら、指中心を開位置 0.125 m、閉位置 0.104 m とし、実際に剛体コライダーとの接触があるか軌跡で確認する。

```sh
scripts/isaac_python.sh "$PWD/runtime/contact_trials.py" --trial grasp \
  --asset "$PWD/my_box.usd" --grasp-left-collider SideL \
  --grasp-right-collider SideR --finger-open-x 0.125 \
  --finger-closed-x 0.104 --finger-center-z 0.07 \
  --report "$PWD/out/my-box-contact.json" --visualizer none
```
