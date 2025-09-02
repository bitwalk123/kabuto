# trading_env_with_features.py
from collections import deque
from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    TradingEnv (改良版)
    - Single-symbol, 1-second tick input (time, price, cumulative_volume)
    - Actions: 0=HOLD, 1=BUY (open long), 2=SELL (open short), 3=REPAY (close)
    - No pyramiding: at most one unit (trade_unit) long OR short
    - Tick size = 1 JPY, slippage = 1 JPY
    - Feature calculation:
        * dvol = np.log1p(delta_cum_volume)
        * MA60, STD60 for price (window=60)
        * RSI60 (simple average-based RSI)
        * Z-score over last 60 prices
    - Warm-up: first `feature_warmup` ticks -> agent must HOLD (env forces HOLD, returns small penalty if action != HOLD)
    - Reward:
        * On REPAY: realized P&L added to reward (profit per share * trade_unit)
        * While holding: give a fraction `hold_reward_frac` of unrealized P&L each tick (can be positive or negative)
        * Illegal actions produce small penalty `punish_illegal`
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        obs_window: int = 120,          # observation length for series (we keep >= feature window)
        feature_window: int = 60,       # window for MA/STD/RSI/Zscore
        trade_unit: int = 100,
        slippage: int = 1,
        tick_size: int = 1,
        hold_reward_frac: float = 0.05, # fraction of unrealized P&L paid each tick while holding
        punish_illegal: float = -1.0,
        reward_scale: float = 1.0,      # optional scaling for rewards (useful for stabilizing)
    ):
        super().__init__()

        assert obs_window >= feature_window, "obs_window must be >= feature_window"

        self.obs_window = obs_window
        self.feature_window = feature_window
        self.trade_unit = int(trade_unit)
        self.slippage = int(slippage)
        self.tick_size = int(tick_size)
        self.hold_reward_frac = float(hold_reward_frac)
        self.punish_illegal = float(punish_illegal)
        self.reward_scale = float(reward_scale)

        # Actions: 0 HOLD, 1 BUY, 2 SELL, 3 REPAY
        self.action_space = spaces.Discrete(4)

        # Observations: dictionary of arrays/scalars
        # - prices: last obs_window prices (right-aligned)
        # - dvol: last obs_window log1p(delta_volume)
        # - ma60, std60, rsi60, zscore60: scalars
        # - pos: scalar in {-1,0,1}
        # - entry_price: scalar
        # - ticks_since_start: scalar (optional)
        self.observation_space = spaces.Dict(
            {
                "prices": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_window,), dtype=np.float32),
                "dvol": spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_window,), dtype=np.float32),
                "ma": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "std": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "rsi": spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32),
                "zscore": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "pos": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
                "entry_price": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
                "ticks_since_start": spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
            }
        )

        # Internal buffers and state
        self._price_buf = deque(maxlen=self.obs_window)
        self._dvol_buf = deque(maxlen=self.obs_window)
        self._cumvol_last = None

        self.position = 0       # -1 short, 0 flat, 1 long
        self.entry_price = 0    # integer price including slippage (JPY)
        self.cum_realized_pnl = 0.0

        self.current_time = None
        self.current_price = None
        self.current_cumvol = None
        self.ticks = 0
        self.feature_warmup = int(self.feature_window)

        self.done = False

    # ---------------- utilities ----------------
    def _to_tick_price(self, price: float) -> int:
        """Round price to nearest tick (tick_size assumed integer)."""
        # careful rounding: convert to nearest tick
        return int(round(price / self.tick_size)) * self.tick_size

    def _get_series_arrays(self):
        p = np.zeros(self.obs_window, dtype=np.float32)
        dv = np.zeros(self.obs_window, dtype=np.float32)
        p_list = list(self._price_buf)
        dv_list = list(self._dvol_buf)
        if len(p_list) > 0:
            p[-len(p_list):] = np.array(p_list, dtype=np.float32)
            dv[-len(dv_list):] = np.array(dv_list, dtype=np.float32)
        return p, dv

    def _calc_ma_std(self, prices: np.ndarray):
        if len(prices) < self.feature_window:
            # fallback: use last price or zeros
            last = prices[-1] if len(prices) > 0 else 0.0
            return float(last), 0.0
        window = prices[-self.feature_window :]
        ma = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        return ma, std

    def _calc_zscore(self, prices: np.ndarray, ma: float, std: float):
        if std == 0 or len(prices) == 0:
            return 0.0
        last = float(prices[-1])
        return float((last - ma) / std)

    def _calc_rsi(self, prices: np.ndarray):
        """
        Simple RSI implementation (period = feature_window)
        Uses average gains/losses (simple moving average) over the period.
        RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss
        If insufficient data: return 50.0 (neutral)
        """
        n = self.feature_window
        if len(prices) < n + 1:
            return 50.0
        window = prices[-(n + 1):]  # need n+1 prices to get n diffs
        deltas = np.diff(window)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.clip(rsi, 0.0, 100.0))

    def _get_observation(self):
        prices, dvol = self._get_series_arrays()
        ma, std = self._calc_ma_std(prices)
        rsi = self._calc_rsi(prices)
        zscore = self._calc_zscore(prices, ma, std)
        obs = {
            "prices": prices.astype(np.float32),
            "dvol": dvol.astype(np.float32),
            "ma": np.array([ma], dtype=np.float32),
            "std": np.array([std], dtype=np.float32),
            "rsi": np.array([rsi], dtype=np.float32),
            "zscore": np.array([zscore], dtype=np.float32),
            "pos": np.array([float(self.position)], dtype=np.float32),
            "entry_price": np.array([float(self.entry_price)], dtype=np.float32),
            "ticks_since_start": np.array([float(self.ticks)], dtype=np.float32),
        }
        return obs

    # ---------------- tick ingestion ----------------
    def ingest_tick(self, t: float, price: float, cumvol: float):
        """
        Ingest external tick: (time, price, cumulative_volume)
        Must be called before step(action) — or step(action, tick=(...)) can be used.
        """
        price_tick = float(self._to_tick_price(price))

        if self._cumvol_last is None:
            dvol = 0.0
        else:
            dvol = float(max(0.0, cumvol - self._cumvol_last))
        self._cumvol_last = float(cumvol)

        self.current_time = float(t)
        self.current_price = float(price_tick)
        self.current_cumvol = float(cumvol)
        self._price_buf.append(float(price_tick))
        # log1p transform for delta volume per spec
        self._dvol_buf.append(float(np.log1p(dvol)))
        self.ticks += 1

    # ---------------- step ----------------
    def step(self, action: int, tick: Optional[Tuple[float, float, float]] = None):
        """
        Apply `action` at current tick. Optionally provide tick tuple to ingest then act.
        Returns: obs, reward, terminated, truncated, info  (gymnasium API)
        """
        if tick is not None:
            self.ingest_tick(*tick)

        if self.current_price is None:
            obs = self._get_observation()
            return obs, 0.0, False, False, {"msg": "no tick yet"}

        info = {}
        reward = 0.0
        illegal = False

        # Enforce warm-up: before feature_warmup ticks, agent must HOLD
        if self.ticks <= self.feature_warmup:
            if action != 0:  # not HOLD
                # force HOLD and penalize slightly to encourage respecting warm-up
                reward += float(self.punish_illegal)
                info["warmup_forced_hold"] = True
                action = 0  # treat it as HOLD for internal logic

        # Action constraints: BUY/SELL only when flat; REPAY only when position != 0
        if action == 1:  # BUY open long
            if self.position != 0:
                illegal = True
            else:
                entry = int(round(self._to_tick_price(self.current_price) + self.slippage))
                self.entry_price = int(entry)
                self.position = 1
                info["entered"] = "long"
                info["entry_price"] = self.entry_price

        elif action == 2:  # SELL open short
            if self.position != 0:
                illegal = True
            else:
                entry = int(round(self._to_tick_price(self.current_price) - self.slippage))
                self.entry_price = int(entry)
                self.position = -1
                info["entered"] = "short"
                info["entry_price"] = self.entry_price

        elif action == 3:  # REPAY close
            if self.position == 0:
                illegal = True
            else:
                # compute exit depending on position direction with slippage
                if self.position == 1:
                    exit_price = int(round(self._to_tick_price(self.current_price) - self.slippage))
                    profit_per_share = exit_price - self.entry_price
                else:  # short
                    exit_price = int(round(self._to_tick_price(self.current_price) + self.slippage))
                    profit_per_share = self.entry_price - exit_price

                realized = float(profit_per_share * self.trade_unit)
                reward += realized  # add realized P&L immediately
                self.cum_realized_pnl += realized
                info["repaid"] = True
                info["exit_price"] = int(exit_price)
                info["realized"] = realized

                # reset position
                self.position = 0
                self.entry_price = 0

        elif action == 0:  # HOLD
            pass
        else:
            illegal = True

        # While holding, give fraction of unrealized P&L as per spec:
        if self.position != 0:
            # compute unrealized P&L per share using current market price and slippage on exit price
            if self.position == 1:
                # long: current exit would be price - slippage
                hypothetical_exit = int(round(self._to_tick_price(self.current_price) - self.slippage))
                unrealized_per_share = hypothetical_exit - self.entry_price
            else:
                hypothetical_exit = int(round(self._to_tick_price(self.current_price) + self.slippage))
                unrealized_per_share = self.entry_price - hypothetical_exit

            unrealized_total = float(unrealized_per_share * self.trade_unit)
            # give only a fraction each tick to stabilize learning
            tick_unrealized_reward = float(self.hold_reward_frac * unrealized_total)
            reward += tick_unrealized_reward
            info["tick_unrealized_reward"] = tick_unrealized_reward
            info["unrealized"] = unrealized_total

        if illegal:
            # treat as HOLD but penalize to teach legal protocol
            reward += float(self.punish_illegal)
            info["illegal_action"] = True

        # optional scaling to keep numeric magnitudes reasonable for NN learning
        reward = float(reward) / max(1.0, float(self.reward_scale))

        obs = self._get_observation()

        terminated = False
        truncated = False

        info.update(
            {
                "step_count": self.ticks,
                "current_time": self.current_time,
                "current_price": self.current_price,
                "cum_realized_pnl": self.cum_realized_pnl,
                "position": self.position,
                "entry_price_internal": self.entry_price,
            }
        )

        return obs, reward, terminated, truncated, info

    # ---------------- reset/render/close ----------------
    def reset(self, seed: Optional[int] = None, options: dict = None):
        super().reset(seed=seed)
        self._price_buf.clear()
        self._dvol_buf.clear()
        self._cumvol_last = None

        self.position = 0
        self.entry_price = 0
        self.cum_realized_pnl = 0.0

        self.current_time = None
        self.current_price = None
        self.current_cumvol = None
        self.ticks = 0
        self.done = False

        return self._get_observation(), {}

    def render(self, mode="human"):
        print(
            f"[t={self.current_time}] price={self.current_price} pos={self.position} "
            f"entry={self.entry_price} cum_realized_pnl={self.cum_realized_pnl}"
        )

    def close(self):
        pass


# ---------------- Example usage ----------------
if __name__ == "__main__":
    env = TradingEnv(obs_window=120, feature_window=60, hold_reward_frac=0.05, reward_scale=1000.0)
    obs, _ = env.reset()

    # simulate 70 ticks so warmup (60) is passed
    rng = np.random.default_rng(0)
    base_price = 1000.0
    cumvol = 0.0
    ticks = []
    for i in range(75):
        # simple random walk for price, and random incremental volume
        base_price += rng.normal(loc=0.0, scale=0.5)
        cumvol += max(0, int(abs(rng.normal(loc=50, scale=20))))
        ticks.append((float(i), float(base_price), float(cumvol)))

    # naive agent: after warmup, buy at tick 61, repay at tick 70
    for i, tick in enumerate(ticks):
        if i == 61:
            action = 1  # BUY
        elif i == 70:
            action = 3  # REPAY
        else:
            action = 0  # HOLD

        obs, reward, terminated, truncated, info = env.step(action, tick=tick)
        print(f"tick={i:03d}, price={tick[1]:.2f}, action={action}, reward={reward:.2f}, pos={info['position']}")

    env.render()
