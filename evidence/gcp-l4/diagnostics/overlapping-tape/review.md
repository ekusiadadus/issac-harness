# 梱包作業台 の画像レビュー

画像と実写を原寸で観察し、具体的な差と判定（合格・要修正）を記入する。

| 条件 | 視点 | PNG SHA-256 | 実写 SHA-256 | 所見・実写との差 | 判定 |
|---|---|---|---|---|---|
| nominal | [overview](nominal/overview.png) | a7808dac455debd6346b9a50745deebe5100817167b5a728141e1c4de055d79e | 97a6789a6c49183c04a006f61d490c5b6117e011345b26682f30af896acdb9b8 | 作業台の脚と床の接地、照明で生じる箱の影を確認。ラベルのある面とフラップが見分けられる。参照写真にある角の潰れ・折れ筋は少ない。 | 合格 |
| nominal | [detail](nominal/detail.png) | 2ec3af256a0c98d691303ec6a2fa38118387eddb726e757162bf79cb38b55cb5 | 88245930f1f148106ded4905a53ee5c655740595f6cf26703fcf8db2b4654aed | 印字と切断面のフルートは読める。右のテープに直線的な三角形の明暗が横切り、角の剥離が一様な淡色のシールに見える。実写の繊維と異なる。 | 要修正 |
| low_overhead | [overview](low_overhead/overview.png) | 962d138a653a4a37ecb377dea8b182752d0bd6cab43cefaa5bbd5f573ff08d4b | 97a6789a6c49183c04a006f61d490c5b6117e011345b26682f30af896acdb9b8 | 脚とコンクリートの輝度は下がるが、端末トレーと箱の影の位置関係は保たれる。実写とは床と照明が異なる。 | 合格 |
| low_overhead | [detail](low_overhead/detail.png) | 7c2ee9fbbf4bb4598f24fe1ae2f4210d2a78abaf94c8add0126b53835a8a3044 | 88245930f1f148106ded4905a53ee5c655740595f6cf26703fcf8db2b4654aed | フラップの下の影が濃くなる。バーコードは読めるが、右端テープの三角境界が照度を変えても残り、スキャン紙面から浮いて見える。 | 要修正 |
