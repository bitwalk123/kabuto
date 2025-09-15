import math
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """
    クラス全体の役割
    - Actor（方策ネットワーク）
      * 観測 obs を入力にして、行動（HOLD / BUY / SELL / REPAY など）の確率分布を出力。
      * PPO では Categorical 分布を使ってサンプリングすることが多い。
    - Critic（価値ネットワーク）
      * 観測 obs を入力にして「その状態の価値（将来の期待リターン）」を推定。
      * PPO の学習では Advantage 計算に利用。

    この ActorCritic クラスは「観測 → 共有特徴量 → (方策出力, 価値出力)」という流れで構成。
    - policy の出力 → アクション選択に使う
    - value の出力 → Advantage 推定に使う
    - 1つのネットワークで両方を同時に学習できるのがポイント
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        """
        :param obs_dim: 観測ベクトルの次元数（特徴量の数）
        :param n_actions: 行動の数（例 HOLD, BUY, SELL, REPAY → 4）
        :param hidden_sizes: 隠れ層のサイズ（デフォルトは 256 → 256）
        """
        super().__init__()
        """
        共有部分（shared layers）
        - obs_dim → 隠れ層 → ReLU → 隠れ層 → ReLU … という順番で積んでいく。
        - 「共有部分」として Actor / Critic の両方で使われる。
        - これにより 状態の特徴表現を学習できる。
        """
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.shared = nn.Sequential(*layers)
        """
        Policy head（方策ネットワーク）
        - 共有表現 shared を入力にして、n_actions 次元の logits を出力。
        - logits は「行動ごとのスコア」で、softmax すれば確率分布になる。
          * 例：[Hold=0.7, Buy=0.2, Sell=0.1]

        1. nn.Linear(last, last//2)
          * 入力次元：last
            これは共有ネットワーク (self.shared) の最後の隠れ層のユニット数。
            例：hidden_sizes=(256,256) なら last=256。
          * 出力次元：last//2
            入力の半分に圧縮している。
            例：256 → 128
          * 役割
            > 「共有特徴」をさらに圧縮して、方策に必要な情報だけを抽出する。
            > Actor 専用の特徴空間を作っている。
        
        2. nn.ReLU()
          * 活性化関数。非線形性を導入することで、複雑な方策関数を表現できる。
          * ReLU は「勾配消失が起きにくい」「計算が速い」ので定番。

        3. nn.Linear(last//2, n_actions)
          * 入力次元：last//2
          * 出力次元：n_actions
          * 役割
            > 各行動に対して「スコア（logit）」を出す。
              例：行動が [HOLD, BUY, SELL, REPAY] なら出力次元 = 4。
            > 出力は確率ではなく logit（softmax 前の値）。
        
        4. 出力の使い方
          * 出力は softmax にかけて確率分布に変換：
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
          * 例：
            logits = [1.2, -0.3, 0.8]
            softmax → [0.55, 0.12, 0.33]
          * この確率分布からアクションをサンプリングする。
        
        まとめ（各パラメータの意味）
        - last : 共有ネットワークの出力次元（観測から抽出された特徴量の数）
        - last//2 : 方策ネットワーク専用の「中間表現の次元」（圧縮して情報抽出）
        - n_actions : 行動空間のサイズ（トレード環境なら 3～4）
        つまり：
        1. 特徴量（last次元） を入力し、
        2. 半分に圧縮（last//2） → ReLU で非線形表現を加える
        3. 行動数分の出力（n_actions） を得て softmax で確率分布を作る
        """
        self.policy = nn.Sequential(
            nn.Linear(last, last // 2),
            nn.ReLU(),
            nn.Linear(last // 2, n_actions)
        )
        """
        Value head（価値ネットワーク）
        - 共有表現 shared を入力にして、スカラー（価値）を出力。
        - 出力は「その状態の良さ（期待される将来の報酬）」を表す。
        
        1. nn.Linear(last, last//2)
          * 入力は 共有特徴量ベクトル（サイズ = last）。
          * 出力は last//2 次元に圧縮。
          → 方策ネットワークと同じく「共有特徴を、価値推定に必要な表現」に変換。
        
        2. nn.ReLU()
          * 非線形性を導入。
          * 方策と同じ理由で「複雑な関数近似を可能にする」。
          
        3. nn.Linear(last//2, 1)
          * 出力は スカラー（次元 = 1）。
          * これが「状態の価値」V(s) の推定値。
          → 方策ヘッドが「確率分布を作る」のに対し、価値ヘッドは「単一の数値（将来の期待リターン）」を出す。
        
        方策ネットワークと価値ネットワークの違い
        - 方策ネットワーク → 「どう動くか？」
        - 価値ネットワーク → 「今の状態はどれだけ良いか？」
        
        なぜ構造が似ているのか？
        - どちらも「状態から特徴を受け取り、それを1つの数値 or 複数のスコアに変換する」タスクだから。
        - 違いは「出力層のサイズと解釈」だけ。
        - 共有部分を self.shared にまとめておくことで、両方が同じ特徴を活用できる。
        
        👉 つまり見た目は同じでも、
        - policy head は「行動を選ぶための分布」を作る
        - value head は「その状態の良さを数値化」する
        という役割分担です。
        """
        self.value = nn.Sequential(
            nn.Linear(last, last // 2),
            nn.ReLU(),
            nn.Linear(last // 2, 1)
        )
        # 重み初期化 (init)
        self.apply(self._weights_init)

    def _weights_init(self, m):
        """
        重み初期化
        - 直交初期化を使って学習の安定性を高めている。
        - PPO は勾配の安定が重要なので、この初期化は有効。
        :param m:
        :return:
        """
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        forward メソッド
        - 入力 x（観測ベクトル）を shared に通す。
        - logits（行動のスコア）と value（状態価値）を同時に返す。
        - PPO の損失関数で両方まとめて学習する。
        :param x:
        :return:
        """
        shared = self.shared(x)
        logits = self.policy(shared)
        value = self.value(shared).squeeze(-1)
        return logits, value
