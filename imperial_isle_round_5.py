from typing import List, Dict
from datamodel import OrderDepth, TradingState, Order, Trade
import numpy as np
import pandas as pd
from collections import defaultdict
import jsonpickle

 
class Trader:

    def __init__(self):

        # === Price History ===
        self.price_history = defaultdict(list)
        self.max_history_len = 60
        self.max_history_len_rock = 169

        # === Generic Bollinger strategy ===
        self.entry_price_map = {}
        self.last_stopped_side_map = {}
        self.must_reverse_map = {}

        self.bollinger_configs = {
            "KELP": {
                "symbol": "KELP",
                "price_source": "mid",
                "window": 7,
                "num_std_dev": 0.2,
                "max_position": 50,
                "max_position_per_tick": 45,
                "stop_loss_pct": 1.0,
                "take_profit_pct": 1.0,
            },
            "SQUID_INK": {
                "symbol": "SQUID_INK",
                "price_source": "mid",
                "window": 15,
                "num_std_dev": 0.75,
                "max_position": 50,
                "max_position_per_tick": 15,
            },
        }

        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "SQUID_INK": 50,
            "KELP": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS":75
        }

        # === VOLCANIC_ROCK options strategy ===
        self.volcanic_lookback = 154
        self.volcanic_z_entry_threshold = 0.8
        self.volcanic_max_position = 400
        self.volcanic_max_position_per_tick = 50
        self.volcanic_stop_loss_pct = 1.0
        self.volcanic_take_profit_pct = 1.0
        self.option_max_position = 200
        self.volcanic_value_threshold = 0.5
        self.av_v10500_purchase = 0
        self.min_pos_rock = 200
        self.volcanic_entry_price = None
        self.volcanic_entry_tick = None
        self.volcanic_must_reverse = False
        self.volcanic_last_stopped_side = None
        self.previous_strategy = None 
        self.rock_tp_1 = False
        self.rock_tp_2 = False
        self.flag = 1

        # === Macarons strat ===
        self.conversions = 0
        self.sun_window = 10
        self.max_position = 75
        self.max_position_per_tick = 75
        self.sunlight_indices = []
        self.tick_counter = 0
        self.entry_price = None
        self.position_side = None
        self.last_position_side = None
        self.take_profit_levels = []
        self.profits_taken = []
        self.previous_gradient = None
        self.entry_tick = None
        self.stop_loss_threshold = -0.1 
        self.arb_max_size = 10

        # === Squid strat ===
        self.extreme_duration = 0
        self.entry_prices = {}
        self.take_profit_pct = {"SQUID_INK": 0.01}

        # === Croissants strat ===
        self.croissants_max_trade_volume = 55

    # -----------------------------------------------------------------------------------------------------------
    # ------------------------------  RUN FUNCTION  ----  TURN STRATEGIES ON AND OFF  ---------------------------
    # -----------------------------------------------------------------------------------------------------------

    def run(self, state: TradingState):
        """
        Main function to run the trader.
        This function is called every tick and should return a list of orders.
        """
        # ---------------------------------------------------------------------
        # 1) Update price history and dynamic variables
        # ---------------------------------------------------------------------
        self._load_state_from_trader_data(state.traderData)  # -- DO NOT DELETE
        self.update_price_history(state)  # -- DO NOT DELETE

        # ---------------------------------------------------------------------
        # 2) Print recent trades & position summary
        # ---------------------------------------------------------------------
        self._get_run_trades(state)  # -- DO NOT DELETE
        self._print_position_summary(state)  # -- DO NOT DELETE

        # ---------------------------------------------------------------------
        # 3) Initialize Order Dictionary
        # ---------------------------------------------------------------------
        orders = {}
        self.conversions=0

        # ----------------------------
        # >>>>>>>>  Kelp Strategies
        # ----------------------------

        # Current Optimal Strategy is Kelp Bollinger Band
        # Kelp Market Maker has been removed for R5

        # >>>> Kelp Bollinger Band <<<<<<
        use_kelp_bollinger = True
        if use_kelp_bollinger:
            kelp_bb_orders = self.run_bollinger_strategy(
                state, self.bollinger_configs["KELP"]
            )
            if kelp_bb_orders:
                if "KELP" in orders:
                    orders["KELP"].extend(kelp_bb_orders)
                else:
                    orders["KELP"] = kelp_bb_orders

        # -----------------------------------------------
        # >>>>>>>>  Volcanic Rock & Options Strategies
        # -----------------------------------------------

        use_volcano_with_options = True
        if use_volcano_with_options:
            volcano_opt_orders = self.run_volcano_with_options_strategy(state)
            for sym, sym_orders in volcano_opt_orders.items():
                if sym not in orders:
                    orders[sym] = []
                orders[sym].extend(sym_orders)

        # --------------------------
        # >>>>>> Squid Strategies
        # --------------------------

        use_squid_strategy = True
        if use_squid_strategy:
            squid_orders = self.run_squid_strategy(state)
            if "SQUID_INK" in orders:
                orders["SQUID_INK"].extend(squid_orders)
            else:
                orders["SQUID_INK"] = squid_orders

        # ----------------------------
        # >>>>>>> Resin Strategies
        # ----------------------------

        # >>>> Rainforest Resin Market Maker <<<<<<
        use_rr_market_taking = True
        if use_rr_market_taking:
            rr_orders = self.run_market_taking_strategy(state, "RAINFOREST_RESIN")
            if "RAINFOREST_RESIN" in orders:
                orders["RAINFOREST_RESIN"].extend(rr_orders)
            else:
                orders["RAINFOREST_RESIN"] = rr_orders

        # >>>> Rainforest Resin Liquidity taker <<<<<<
        use_rr_fixed_price = True
        if use_rr_fixed_price:
            fp_rr_orders = self.run_fixed_price_strategy(state, "RAINFOREST_RESIN")
            if "RAINFOREST_RESIN" in orders:
                orders["RAINFOREST_RESIN"].extend(fp_rr_orders)
            else:
                orders["RAINFOREST_RESIN"] = fp_rr_orders

        # -----------------------------
        # >>>>>>> DJEMBE Strategies
        # -----------------------------

        # >>>> DJEMBE NEW Strategy Band <<<<<<
        use_djembe_rolling = True
        if use_djembe_rolling:
            djembe_orders = self.run_djembes_strategy(state)
            if "DJEMBES" in orders:
                orders["DJEMBES"].extend(djembe_orders)
            else:
                orders["DJEMBES"] = djembe_orders

        # ---------------------------------
        # >>>>>>>>  Croissants Strategy
        # ---------------------------------
        use_croissants_strategy = True
        if use_croissants_strategy:
            croissants_orders = self.run_croissants_strategy(state)
            if "CROISSANTS" in orders:
                orders["CROISSANTS"].extend(croissants_orders)
            else:
                orders["CROISSANTS"] = croissants_orders

        # -------------------------------
        # >>>>>>>>  Picnic Strategies
        # -------------------------------

        # Arbitrage between baskets and components
        use_picnic_index_arbitrage = True
        if use_picnic_index_arbitrage:
            arb_orders = self.run_picnic_index_arbitrage(state)
            for sym, sym_orders in arb_orders.items():
                if sym not in orders:
                    orders[sym] = []
                orders[sym].extend(sym_orders)
                
        use_mac = True
        if use_mac:
            mac_orders = self.run_macaron_strategy(state)
            for sym, sym_orders in mac_orders.items():
                if sym not in orders:
                    orders[sym] = []
                orders[sym].extend(sym_orders)

        traderData = self._save_state_to_trader_data()

        return orders, self.conversions, traderData

    # -----------------------------------------------------------------------------------------------------------
    # --------------------------------  SAVING, LOADING, AND UPDATING PRICE HISTORY  ----------------------------
    # -----------------------------------------------------------------------------------------------------------

    def update_price_history(self, state: TradingState):
        for product, order_depth in state.order_depths.items():
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders)
                best_ask = min(order_depth.sell_orders)
                mid_price = (best_bid + best_ask) / 2
                self.price_history[product].append(mid_price)
                 # Custom history length
                if product == "VOLCANIC_ROCK":
                    self.price_history[product] = self.price_history[product][-self.max_history_len_rock:]
                else:
                    self.price_history[product] = self.price_history[product][-self.max_history_len:]

    def _load_state_from_trader_data(self, trader_data: str):
        if trader_data:
            try:
                state = jsonpickle.decode(trader_data)
                for k, v in state.items():
                    setattr(self, k, v)
            except Exception as e:
                print("[ERROR] Failed to load state from traderData:", e)

    def _save_state_to_trader_data(self) -> str:
        state_vars = {

            "price_history": self.price_history,

            # MACARON STRATEGY
            "entry_price": self.entry_price,
            "entry_tick": self.entry_tick,
            "position_side": self.position_side,
            "last_position_side": self.last_position_side,
            "take_profit_levels": self.take_profit_levels,
            "profits_taken": self.profits_taken,
            "previous_gradient": self.previous_gradient,
            "sunlight_indices": self.sunlight_indices,

            # VOLCANO STRATEGY
            "volcanic_entry_price": self.volcanic_entry_price,
            "volcanic_entry_tick": self.volcanic_entry_tick,
            "volcanic_must_reverse": self.volcanic_must_reverse,
            "volcanic_last_stopped_side": self.volcanic_last_stopped_side,
            "min_pos_rock": self.min_pos_rock,
            "rock_tp_1": self.rock_tp_1,
            "rock_tp_2": self.rock_tp_2,

            # SQUID STRATEGY
            "extreme_duration": self.extreme_duration,
            "entry_prices": self.entry_prices,
            "take_profit_pct": self.take_profit_pct,

            # VRC105 LOGIC
            "av_v10500_purchase": self.av_v10500_purchase,
            "flag": self.flag,

        }

        try:
            return jsonpickle.encode(state_vars)
        except Exception as e:
            print("[ERROR] Failed to save state:", e)
            return ""

    # -----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------  DEBUGGING  ------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    def _get_run_trades(self, state: TradingState) -> List[Trade]:
        """
        Retrieve and print trades from the previous run (timestamp - 100).
        Format: [TRD-B-3x-RF-9998]
        """
        trades = []
        current_ts = state.timestamp - 100
        formatted_trades = []

        for product, trade_list in state.own_trades.items():
            if product not in self.price_history:
                continue

            recent_trades = [t for t in trade_list if t.timestamp == current_ts]
            trades.extend(recent_trades)

            short_symbol = self._short_symbol(product)

            for trade in recent_trades:
                direction = "B" if trade.buyer == "SUBMISSION" else "S"
                formatted_trades.append(
                    f"[TRD-{direction}-{abs(trade.quantity)}x-{short_symbol}-{trade.price}]"
                )

        if formatted_trades:
            print(f"[TRADES @ t={current_ts}] {' '.join(formatted_trades)}")
        else:
            print(f"[TRADES @ t={current_ts}] None")

        return trades

    def _short_symbol(self, symbol: str) -> str:
        """
        Convert product name to a short symbol for compact logging.
        Example: RAINFOREST_RESIN -> RR, CROISSANTS -> C
        """
        shortcuts = {
            "RAINFOREST_RESIN": "RR",
            "SQUID_INK": "SI",
            "KELP": "K",
            "CROISSANTS": "C",
            "JAMS": "J",
            "DJEMBES": "D",
            "PICNIC_BASKET1": "PB1",
            "PICNIC_BASKET2": "PB2",
            "VOLCANIC_ROCK": "VR",
            "VOLCANIC_ROCK_VOUCHER_9500": "VRC95",
            "VOLCANIC_ROCK_VOUCHER_9750": "VRC97",
            "VOLCANIC_ROCK_VOUCHER_10000": "VRC100",
            "VOLCANIC_ROCK_VOUCHER_10250": "VRC102",
            "VOLCANIC_ROCK_VOUCHER_10500": "VRC105",
            "MAGNIFICENT_MACARONS":"MM"
        }
        return shortcuts.get(symbol, symbol[:2])

    def _print_position_summary(self, state: TradingState):
        """
        Prints the current position for all known products in a compact format.
        Example: [POS-RF:+5] [POS-SI:-2]
        """
        positions = []
        for product, pos in state.position.items():
            if product not in self.price_history:
                continue
            short_symbol = self._short_symbol(product)
            sign = "+" if pos >= 0 else ""
            positions.append(f"[POS-{short_symbol}:{sign}{pos}]")

        if positions:
            print(" ".join(positions))
        else:
            print("[POS] No positions")

    # -----------------------------------------------------------------------------------------------------------
    # -------------------------------------------  ORDER STRUCTURE  ---------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    def limit_order(self, symbol: str, price: float, quantity: int, side: str) -> Order:
        """
        Function to create and return a limit order.
        Prints a compact log like [LO-B-5x-RF-10000]
        """
        if side.lower() == "buy":
            signed_quantity = abs(quantity)
            tag = "B"
        elif side.lower() == "sell":
            signed_quantity = -abs(quantity)
            tag = "S"
        else:
            raise ValueError(f"Invalid order side '{side}' — must be 'buy' or 'sell'")

        short_symbol = self._short_symbol(symbol)
        print(f"[LO-{tag}-{abs(signed_quantity)}x-{short_symbol}-{price}]")
        return Order(symbol=symbol, price=price, quantity=signed_quantity)

    def get_best_bid_ask(self, state: TradingState, symbol: str) -> tuple[float, float]:
        """
        Retrieve the best bid and ask prices for a given symbol.
        Returns: (best_bid, best_ask)
        """
        order_book = state.order_depths[symbol]
        best_bid = max(order_book.buy_orders) if order_book.buy_orders else 0.0
        best_ask = (
            min(order_book.sell_orders) if order_book.sell_orders else float("inf")
        )
        return best_bid, best_ask

    def market_order(
        self, state: TradingState, symbol: str, quantity: int, side: str
    ) -> Order:
        """
        Function to create and return a market order using the best bid/ask price.
        Prints a compact log like [MO-B-5x-RF-10000]
        """
        best_bid, best_ask = self.get_best_bid_ask(state, symbol)

        if side.lower() == "buy":
            price = best_ask
            signed_quantity = abs(quantity)
            tag = "B"
        elif side.lower() == "sell":
            price = best_bid
            signed_quantity = -abs(quantity)
            tag = "S"
        else:
            raise ValueError(f"Invalid order side '{side}' — must be 'buy' or 'sell'")

        short_symbol = self._short_symbol(symbol)
        print(f"[MO-{tag}-{abs(signed_quantity)}x-{short_symbol}-{price}]")
        return Order(symbol=symbol, price=price, quantity=signed_quantity)
    
    # -----------------------------------------------------------------------------------------------------------
    # ------------------------------------  GENERIC/UNIVERSAL STRATEGIES  ---------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    def calculate_ema(self, prices, window):

        if not prices:
            return 0
        if len(prices) < window:
            return sum(prices) / len(prices)
        multiplier = 2 / (window + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema

    def compute_bollinger_bands(self, prices, window=20, num_std_dev=2.0):
        if len(prices) < window:
            return None
        series = pd.Series(prices[-window:])
        ma = series.mean()
        std = series.std()
        upper = ma + num_std_dev * std
        lower = ma - num_std_dev * std
        return ma, upper, lower

    def run_bollinger_strategy(self, state: TradingState, config: dict) -> List[Order]:
        symbol = config["symbol"]
        orders: List[Order] = []

        if symbol not in state.order_depths:
            return orders

        order_depth = state.order_depths[symbol]
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return orders

        if config.get("price_source") == "sampled":
            sampled_prices = self.price_history[symbol][::25][-config["window"] :]
            if len(sampled_prices) < config["window"]:
                return []
            price_history = sampled_prices
        else:
            if len(self.price_history[symbol]) < config["window"]:
                return []
            price_history = self.price_history[symbol]

        bands = self.compute_bollinger_bands(
            price_history, config["window"], config["num_std_dev"]
        )
        if bands is None:
            return []

        ma, upper, lower = bands

        position = state.position.get(symbol, 0)
        entry_price = self.entry_price_map.get(symbol)
        must_reverse = self.must_reverse_map.get(symbol, False)
        last_stopped = self.last_stopped_side_map.get(symbol)

        if entry_price is not None and position != 0:
            tp = (
                entry_price * (1 + config["take_profit_pct"])
                if position > 0
                else entry_price * (1 - config["take_profit_pct"])
            )
            sl = (
                entry_price * (1 - config["stop_loss_pct"])
                if position > 0
                else entry_price * (1 + config["stop_loss_pct"])
            )
            if (position > 0 and (best_bid >= tp or best_bid <= sl)) or (
                position < 0 and (best_ask <= tp or best_ask >= sl)
            ):
                volume = abs(position)
                side = "sell" if position > 0 else "buy"
                px = best_bid if position > 0 else best_ask
                orders.append(self.limit_order(symbol, px, volume, side))
                self.entry_price_map[symbol] = None
                self.must_reverse_map[symbol] = True
                self.last_stopped_side_map[symbol] = "long" if position > 0 else "short"
                return orders

        if best_ask < lower and position < config["max_position"]:
            if must_reverse and last_stopped == "long":
                return orders
            volume = min(
                config["max_position"] - position,
                abs(order_depth.sell_orders[best_ask]),
                config["max_position_per_tick"],
            )
            if volume > 0:
                orders.append(self.limit_order(symbol, best_ask, volume, "buy"))
                self.entry_price_map[symbol] = best_ask
                self.must_reverse_map[symbol] = False

        elif best_bid > upper and position > -config["max_position"]:
            if must_reverse and last_stopped == "short":
                return orders
            volume = min(
                config["max_position"] + position,
                abs(order_depth.buy_orders[best_bid]),
                config["max_position_per_tick"],
            )
            if volume > 0:
                orders.append(self.limit_order(symbol, best_bid, volume, "sell"))
                self.entry_price_map[symbol] = best_bid
                self.must_reverse_map[symbol] = False

        return orders
    
    # -----------------------------------------------------------------------------------------------------------
    # ------------------------------------------  CUSTOM STRATEGIES  --------------------------------------------
    # -----------------------------------------------------------------------------------------------------------

    def run_volcano_with_options_strategy(
        self, state: TradingState
    ) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        rock_product = "VOLCANIC_ROCK"

        option_products = [
            p for p in state.order_depths if p.startswith("VOLCANIC_ROCK_VOUCHER_")
        ]

        if rock_product not in state.order_depths:
            return result

        order_depth = state.order_depths[rock_product]
        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders

        if not buy_orders or not sell_orders:
            return result

        # Collect best bid/ask for the underlying
        best_bid = max(buy_orders)
        best_ask = min(sell_orders)
        best_bid_volume = abs(buy_orders[best_bid])
        best_ask_volume = abs(sell_orders[best_ask])

        # Use shared price history system
        prices = self.price_history.get(rock_product, [])
        if len(prices) < (self.volcanic_lookback + 10):
            return result

        # Use the underlyings unique lookback window
        recent_prices = prices[-(self.volcanic_lookback + 10) :]
        df = pd.DataFrame({"mid_price": recent_prices})

        df["ma"] = df["mid_price"].rolling(window=self.volcanic_lookback).mean()
        df["std"] = df["mid_price"].rolling(window=self.volcanic_lookback).std()
        df["upper"] = df["ma"] + self.volcanic_z_entry_threshold * df["std"]
        df["lower"] = df["ma"] - self.volcanic_z_entry_threshold * df["std"]

        upper = df["upper"].iloc[-1]
        lower = df["lower"].iloc[-1]

        orders: List[Order] = []
        rock_position = state.position.get(rock_product, 0)

        # Stop loss / take profit logic -> Currently Redundant 
        if self.volcanic_entry_price is not None and rock_position != 0:
            if rock_position > 0:
                tp = self.volcanic_entry_price * (1 + self.volcanic_take_profit_pct)
                sl = self.volcanic_entry_price * (1 - self.volcanic_stop_loss_pct)
                if best_bid >= tp or best_bid <= sl:
                    orders.append(Order(rock_product, best_bid, -abs(rock_position)))
                    self.volcanic_entry_price = None
                    self.volcanic_must_reverse = True
                    self.volcanic_last_stopped_side = "long"
            elif rock_position < 0:
                tp = self.volcanic_entry_price * (1 - self.volcanic_take_profit_pct)
                sl = self.volcanic_entry_price * (1 + self.volcanic_stop_loss_pct)
                if best_ask <= tp or best_ask >= sl:
                    orders.append(Order(rock_product, best_ask, abs(rock_position)))
                    self.volcanic_entry_price = None
                    self.volcanic_must_reverse = True
                    self.volcanic_last_stopped_side = "short"

        # Entry signal
        rock_signal = None
        if best_ask < lower:
            rock_signal = "buy"
        elif best_bid > upper:
            rock_signal = "sell"

        if rock_signal == "buy" and rock_position < self.volcanic_max_position:
            qty = min(
                self.volcanic_max_position - rock_position,
                best_ask_volume,
                self.volcanic_max_position_per_tick,
            )
            orders.append(Order(rock_product, best_ask, qty))
            self.volcanic_entry_price = best_ask
            self.volcanic_must_reverse = False

        elif rock_signal == "sell" and rock_position > -self.volcanic_max_position:
            qty = min(
                self.volcanic_max_position + rock_position,
                best_bid_volume,
                self.volcanic_max_position_per_tick,
            )
            orders.append(Order(rock_product, best_bid, -qty))
            self.volcanic_entry_price = best_bid
            self.volcanic_must_reverse = False

        # Iterate through underlying contracts
        for opt in option_products:
            try:
                strike = int(opt.split("_")[-1])
            except:
                continue

            opt_depth = state.order_depths[opt]
            opt_position = state.position.get(opt, 0)
            intrinsic = best_ask - strike

            # Just for final round, ignore options strategy for furthest OTM calls only
            # Immediately purchase 200 10500 strike calls at $1; Hail Mary play
            if self.flag == 1:
                if opt == "VOLCANIC_ROCK_VOUCHER_10500":
                    self.min_pos_rock = 200
                    opt_ask_v10500 = (
                        min(opt_depth.sell_orders) if opt_depth.sell_orders else None
                    )
                    opt_volume_v10500 = min(
                        (
                            abs(opt_depth.sell_orders[opt_ask_v10500])
                            if opt_ask_v10500
                            else 0
                        ),
                        (self.min_pos_rock - state.position.get(opt, 0)),
                    )
                    orders.append(Order(opt, opt_ask_v10500, opt_volume_v10500))
                    self.av_v10500_purchase += opt_ask_v10500 * opt_volume_v10500
                    if state.position.get(opt, 0) >= 200:
                        self.flag = 0
                        self.av_v10500_purchase = (
                            self.av_v10500_purchase / state.position.get(opt, 0)
                        )

            # TP logic on 10500 Calls
            if self.av_v10500_purchase > 0:
                if opt == "VOLCANIC_ROCK_VOUCHER_10500":
                    opt_bid = (
                        max(opt_depth.buy_orders) if opt_depth.buy_orders else None
                    )
                    if opt_bid is not None and self.rock_tp_1 == False:
                        if opt_bid >= self.av_v10500_purchase * 33:
                            self.min_pos_rock = 133
                            self.rock_tp_1 = True
                    if opt_bid is not None and self.rock_tp_2 == False:
                        if opt_bid >= self.av_v10500_purchase * 50:
                            self.min_pos_rock = 66
                            self.rock_tp_2 = True
                    if opt_bid is not None:
                        if opt_bid >= self.av_v10500_purchase * 66:
                            self.min_pos_rock = 0
                    opt_volume = (
                        max(state.position.get(opt, 0) - self.min_pos_rock, 0) if opt_bid else 0
                    )

                    if opt_bid and intrinsic <= opt_bid * (
                        1 + self.volcanic_value_threshold
                    ):
                        qty = min(
                            self.option_max_position + opt_position,
                            opt_volume,
                            self.volcanic_max_position_per_tick,
                        )
                        if qty > 0:
                            orders.append(Order(opt, opt_bid, -qty))

            # Trade strategy on all ITM options as most correlated with underlying
            # Don't continue if OTM and existing position as must be able to exit
            if intrinsic <= 0 and opt_position == 0:
                continue  # ITM only

            if opt == "VOLCANIC_ROCK_VOUCHER_10500":
                continue  # Don't trade strategy on 10500s

            if rock_signal == "buy":
                opt_ask = min(opt_depth.sell_orders) if opt_depth.sell_orders else None
                opt_volume = abs(opt_depth.sell_orders[opt_ask]) if opt_ask else 0

                if opt_ask and intrinsic >= opt_ask * (
                    1 - self.volcanic_value_threshold
                ):
                    qty = min(
                        self.option_max_position - opt_position,
                        opt_volume,
                        self.volcanic_max_position_per_tick,
                    )
                    if qty > 0:
                        orders.append(Order(opt, opt_ask, qty))

            elif rock_signal == "sell":
                opt_bid = (
                    max(opt_depth.buy_orders) if opt_depth.buy_orders else None
                )
                opt_volume = abs(opt_depth.buy_orders[opt_bid]) if opt_bid else 0
                if opt_bid and intrinsic <= opt_bid * (
                    1 + self.volcanic_value_threshold
                ):
                    qty = min(
                        self.option_max_position + opt_position,
                        opt_volume,
                        self.volcanic_max_position_per_tick,
                    )
                    if qty > 0:
                        orders.append(Order(opt, opt_bid, -qty))

        # Return orders
        for order in orders:
            result.setdefault(order.symbol, []).append(order)

        return result


    def run_market_taking_strategy(
        self, state: TradingState, product: str
    ) -> List[Order]:
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        acceptable_price = 10000
        position = state.position.get(product, 0)

        max_position = 50
        min_position = -50

        # Take cheap asks
        for ask in sorted(order_depth.sell_orders.keys()):
            if ask > acceptable_price:
                continue

            ask_volume = order_depth.sell_orders[ask]
            available_buy_capacity = max_position - position
            if available_buy_capacity <= 0:
                break

            buy_quantity = min(available_buy_capacity, abs(ask_volume))
            orders.append(self.market_order(state, product, buy_quantity, "buy"))
            position += buy_quantity

        # Take rich bids
        for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid < acceptable_price:
                continue

            bid_volume = order_depth.buy_orders[bid]
            available_sell_capacity = position - min_position
            if available_sell_capacity <= 0:
                break

            sell_quantity = min(available_sell_capacity, bid_volume)

            orders.append(self.market_order(state, product, sell_quantity, "sell"))
            position -= sell_quantity

        return orders

    def run_fixed_price_strategy(
        self, state: TradingState, product: str
    ) -> List[Order]:
        position = state.position.get(product, 0)
        orders: List[Order] = []

        limit = 40
        fixed_qty = 9

        normal_buy_price = 9998
        normal_sell_price = 10002
        unload_sell_price = 9999
        unload_buy_price = 10001

        if -limit <= position <= limit:
            orders.append(self.limit_order(product, normal_buy_price, fixed_qty, "buy"))
            orders.append(
                self.limit_order(product, normal_sell_price, fixed_qty, "sell")
            )
        elif position > limit:
            unload_qty = min(position - limit, 8)
            orders.append(
                self.limit_order(product, unload_sell_price, unload_qty, "sell")
            )
        elif position < -limit:
            unload_qty = min(-limit - position, 8)
            orders.append(
                self.limit_order(product, unload_buy_price, unload_qty, "buy")
            )

        return orders

    def run_croissants_strategy(self, state: TradingState) -> List[Order]:

        product = "CROISSANTS"
        if product not in state.order_depths or product not in state.market_trades:
            return []

        trades = [t for t in state.market_trades[product] if t.buyer == "Olivia" or t.seller == "Olivia"]
        if not trades:
            return []

        position = state.position.get(product, 0)
        limit = self.position_limits[product]
        order_depth = state.order_depths[product]
        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else float("inf")
        
        orders: List[Order] = []

        for trade in trades:
            if trade.buyer == "Olivia":
                qty = min(limit - position, self.croissants_max_trade_volume)
                if best_ask < float("inf") and qty > 0:
                    orders.append(self.limit_order(product, best_ask, qty, "buy"))
                    position += qty
            elif trade.seller == "Olivia":
                qty = min(position + limit, self.croissants_max_trade_volume)
                if best_bid > 0 and qty > 0:
                    orders.append(self.limit_order(product, best_bid, qty, "sell"))
                    position -= qty

        return orders

    def run_picnic_index_arbitrage(self, state: TradingState) -> Dict[str, List[Order]]:
        """ "
        Arbitrage strategy comparing the actual price of each PICNIC_BASKET
        to the fair value derived from its underlying components.

        Composition assumptions:
          - PICNIC_BASKET1 = 6 CROISSANTS + 3 JAMS + 1 DJEMBE
          - PICNIC_BASKET2 = 4 CROISSANTS + 2 JAMS

        Steps:
          1) Retrieve best bid/ask for each relevant component (CROISSANTS, JAMS, DJEMBES)
          2) Compute mid-prices => theoretical fair cost for each basket
          3) Retrieve actual basket mid-price
          4) Compare (actual - theoretical) => if difference > threshold => short basket, buy components.
             If difference < -threshold => buy basket, short components.
        """
        orders: Dict[str, List[Order]] = {
            "PICNIC_BASKET1": [],
            "PICNIC_BASKET2": [],
            # We also may place orders in underlying components
            "CROISSANTS": [],
            "JAMS": [],
            "DJEMBES": [],
        }

        # --------------------------
        #  Retrieve mid-prices
        # --------------------------
        # Utility to safely get mid-price or None
        def get_mid(symbol: str):
            if symbol not in state.order_depths:
                return None
            b, a = self.get_best_bid_ask(state, symbol)
            if a == float("inf"):
                return None
            return 0.5 * (b + a)

        # -- For PB1
        pb1_mid = get_mid("PICNIC_BASKET1")
        cr_mid = get_mid("CROISSANTS")
        jam_mid = get_mid("JAMS")
        dj_mid = get_mid("DJEMBES")

        # -- For PB2
        pb2_mid = get_mid("PICNIC_BASKET2")

        # If we can't price everything, skip
        # (You can refine to skip only PB1 or only PB2, if partial data is missing.)
        if (
            pb1_mid is None
            or cr_mid is None
            or jam_mid is None
            or dj_mid is None
            or pb2_mid is None
        ):
            return orders

        # --------------------------
        #  Theoretical cost
        # --------------------------
        # PB1 = 6C + 3J + 1D
        fair_pb1 = 6 * cr_mid + 3 * jam_mid + 1 * dj_mid
        # PB2 = 4C + 2J
        fair_pb2 = 4 * cr_mid + 2 * jam_mid

        # Compare actual to theoretical
        diff_pb1 = pb1_mid - fair_pb1
        diff_pb2 = pb2_mid - fair_pb2

        # Threshold to avoid tiny mispricings; tune as needed
        threshold_1 = 30.0
        threshold_2 = 23.0

        # We'll do a small trade size if we see mispricing
        # In a real scenario, you'd also check your position & limit
        trade_size_pb1 = 2
        trade_size_pb2 = 2

        # ---------------------------------------
        # PB1 Arbitrage
        # ---------------------------------------
        if diff_pb1 > threshold_1:
            # PB1 is overpriced => short PB1, buy components
            # Sell basket
            best_bid_pb1, _ = self.get_best_bid_ask(state, "PICNIC_BASKET1")
            if best_bid_pb1 > 0:
                orders["PICNIC_BASKET1"].append(
                    self.limit_order(
                        "PICNIC_BASKET1", best_bid_pb1, trade_size_pb1, "sell"
                    )
                )

            # Buy underlying in proportion => 6C, 3J, 1D
            # We'll just do an example single-lot for each if you're truly matching notional
            # you'd compute the ratio. For simplicity, let's place small partial orders:

            best_ask_jam, _ = self.get_best_bid_ask(state, "JAMS")

            # Make sure they're not inf
            if best_ask_jam < float("inf"):
                orders["JAMS"].append(
                    self.limit_order("JAMS", best_ask_jam, 3 * trade_size_pb1, "buy")
                )

        elif diff_pb1 < -threshold_1:
            # PB1 is underpriced => buy PB1, short components
            # Buy basket
            _, best_ask_pb1 = self.get_best_bid_ask(state, "PICNIC_BASKET1")
            if best_ask_pb1 < float("inf"):
                orders["PICNIC_BASKET1"].append(
                    self.limit_order(
                        "PICNIC_BASKET1", best_ask_pb1, trade_size_pb1, "buy"
                    )
                )

            # Short the underlying

            best_bid_jam, _ = self.get_best_bid_ask(state, "JAMS")

            if best_bid_jam > 0:
                orders["JAMS"].append(
                    self.limit_order("JAMS", best_bid_jam, 3 * trade_size_pb1, "sell")
                )

        # ---------------------------------------
        # PB2 Arbitrage
        # ---------------------------------------
        if diff_pb2 > threshold_2:
            # PB2 overpriced => short PB2, buy the components (4C + 2J)
            best_bid_pb2, _ = self.get_best_bid_ask(state, "PICNIC_BASKET2")
            if best_bid_pb2 > 0:
                orders["PICNIC_BASKET2"].append(
                    self.limit_order(
                        "PICNIC_BASKET2", best_bid_pb2, trade_size_pb2, "sell"
                    )
                )

            best_ask_jam, _ = self.get_best_bid_ask(state, "JAMS")
            if best_ask_jam < float("inf"):
                orders["JAMS"].append(
                    self.limit_order("JAMS", best_ask_jam, 2 * trade_size_pb2, "buy")
                )

        elif diff_pb2 < -threshold_2:
            # PB2 underpriced => buy PB2, short its components
            _, best_ask_pb2 = self.get_best_bid_ask(state, "PICNIC_BASKET2")
            if best_ask_pb2 < float("inf"):
                orders["PICNIC_BASKET2"].append(
                    self.limit_order(
                        "PICNIC_BASKET2", best_ask_pb2, trade_size_pb2, "buy"
                    )
                )

            best_bid_jam, _ = self.get_best_bid_ask(state, "JAMS")

            if best_bid_jam > 0:
                orders["JAMS"].append(
                    self.limit_order("JAMS", best_bid_jam, 2 * trade_size_pb2, "sell")
                )

        return orders

    def run_djembes_strategy(self, state: TradingState) -> List[Order]:
        product = "DJEMBES"
        if product not in state.order_depths:
            return []

        # Get market data
        order_depth = state.order_depths[product]
        best_bid = (
            max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        )
        best_ask = (
            min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        )

        mid_price = (best_bid + best_ask) / 2
        current_pos = state.position.get(product, 0)
        pos_limit = self.position_limits[product]  # 250

        # Calculate fair value using EMA (more responsive than SMA)
        price_history = self.price_history.get(product, [])

        # Calculate EMAs
        ema_short = self.calculate_ema(price_history[-3:], 3)  # 5-period EMA
        ema_long = self.calculate_ema(price_history[-15:], 15)  # 20-period EMA
        momentum = (
            price_history[-1] - price_history[-3] if len(price_history) >= 3 else 0
        )
        fair_value = (
            ema_short + ema_long
        ) / 2 + momentum * 0.34

        # Calculate volatility (standard deviation)
        std_dev = np.std(price_history[-15:]) if len(price_history) >= 15 else 1.8 

        # Dynamic spread calculation
        base_spread = max(0.5, std_dev * 0.8)  # Minimum spread of 0.5   
        spread_adjustment = 1 + (
            abs(current_pos) / pos_limit * 1.5
        )  # Wider spread as position increases

        # Calculate bid/ask prices
        bid_price = round(fair_value - (base_spread * spread_adjustment))
        ask_price = round(fair_value + (base_spread * spread_adjustment))

        # Dynamic position-based order sizing
        base_size = 10  # Base order size
        size_reduction = int(base_size * (abs(current_pos) / pos_limit))
        order_size = max(5, base_size - size_reduction)  # Never go below 5

        orders = []

        # Place bids if we're not at max position
        if current_pos < pos_limit:
            orders.append(self.limit_order(product, bid_price, order_size, "buy"))

        # Place asks if we're not at min position
        if current_pos > -pos_limit:
            orders.append(self.limit_order(product, ask_price, order_size, "sell"))

        # Mean reversion logic for large positions
        if current_pos > pos_limit * 0.7:  # If long too much
            if mid_price > fair_value + std_dev * 0.4:  # Price above fair value
                unload_size = min(5, current_pos - int(pos_limit * 0.3))
                orders.append(self.limit_order(product, best_bid, unload_size, "sell"))

        elif current_pos < -pos_limit * 0.7:  # If short too much
            if mid_price < fair_value - std_dev * 0.4:  # Price below fair value
                unload_size = min(5, abs(current_pos) - int(pos_limit * 0.3))
                orders.append(self.limit_order(product, best_ask, unload_size, "buy"))

        # Aggressive mean reversion at extreme prices
        if mid_price > fair_value + std_dev * 1.5:  # 1.5 std dev above mean
            if current_pos > -pos_limit:
                sell_size = min(10, pos_limit + current_pos)
                orders.append(self.limit_order(product, ask_price, sell_size, "sell"))

        elif mid_price < fair_value - std_dev * 1.5:  # 1.5 std dev below mean
            if current_pos < pos_limit:
                buy_size = min(10, pos_limit - current_pos)
                orders.append(self.limit_order(product, bid_price, buy_size, "buy"))

        return orders

    def run_squid_strategy(self, state: TradingState) -> List[Order]:
        symbol = "SQUID_INK"
        orders: List[Order] = []

        if symbol not in state.order_depths or symbol not in self.price_history:
            return orders
        if len(self.price_history[symbol]) < 55:
            return orders
        prices = self.price_history[symbol]

        rolling_avg = np.mean(prices[-50:])
        current_price = prices[-1]
        deviation = current_price - rolling_avg

        if not hasattr(self, "extreme_duration"):
            self.extreme_duration = 0
        if not hasattr(self, "entry_prices"):
            self.entry_prices = {}
        if not hasattr(self, "take_profit_pct"):
            self.take_profit_pct = {symbol: 0.01}

        # Fetch market data
        order_depth = state.order_depths[symbol]
        best_bid, best_ask = self.get_best_bid_ask(state, symbol)
        position = state.position.get(symbol, 0)
        max_pos = self.bollinger_configs[symbol]["max_position"]

        # --- Exit strategy: Take profit if target reached ---
        entry_price = self.entry_prices.get(symbol)
        tp_pct = self.take_profit_pct[symbol]
        # Long take-profit
        if position > 0 and entry_price is not None:
            if current_price >= entry_price * (1 + tp_pct):
                vol = position
                orders.append(self.limit_order(symbol, best_bid, vol, "sell"))
                # Reset entry price after exit
                del self.entry_prices[symbol]
                return orders
        # Short take-profit
        if position < 0 and entry_price is not None:
            if current_price <= entry_price * (1 - tp_pct):
                vol = abs(position)
                orders.append(self.limit_order(symbol, best_ask, vol, "buy"))
                del self.entry_prices[symbol]
                return orders

        # Define thresholds
        extreme_threshold = 40
        trigger_duration = 1

        if deviation >= extreme_threshold:
            if extreme_threshold >= 50:
                self.bollinger_configs[symbol]["max_position_per_tick"] = 25
            self.extreme_duration += 1
            if self.extreme_duration >= trigger_duration and position > -max_pos:
                volume = min(
                    max_pos + position,
                    self.bollinger_configs[symbol]["max_position_per_tick"],
                    abs(order_depth.buy_orders.get(best_bid, 0)),
                )
                if volume > 0:
                    # Overextended: enter short
                    orders.append(self.limit_order(symbol, best_bid, volume, "sell"))
                    # Record entry price
                    self.entry_prices[symbol] = best_bid
                    self.extreme_duration = 0
                    self.bollinger_configs[symbol]["max_position_per_tick"] = 7

        elif deviation <= -extreme_threshold:
            if abs(extreme_threshold) >= 50:
                self.bollinger_configs[symbol]["max_position_per_tick"] = 25
            self.extreme_duration += 1
            if self.extreme_duration >= trigger_duration and position < max_pos:
                volume = min(
                    max_pos - position,
                    self.bollinger_configs[symbol]["max_position_per_tick"],
                    abs(order_depth.sell_orders.get(best_ask, 0)),
                )
                if volume > 0:
                    # Oversold: enter long
                    orders.append(self.limit_order(symbol, best_ask, volume, "buy"))
                    self.entry_prices[symbol] = best_ask
                    self.extreme_duration = 0
                    self.bollinger_configs[symbol]["max_position_per_tick"] = 7
        else:
            # Reset if no longer extreme
            self.extreme_duration = 0

        for trade in state.market_trades.get(symbol, []):
            # how much capacity to go long
            if trade.buyer == "Olivia":
                cap = max_pos - position
                qty = min(cap, trade.quantity)
                if qty > 0 and best_ask != float("inf"):
                    orders.append(self.limit_order(symbol, best_ask, qty, "buy"))
                    position += qty

            # how much capacity to go short
            elif trade.seller == "Olivia":
                cap = position + max_pos
                qty = min(cap, trade.quantity)
                if qty > 0 and best_bid > 0:
                    orders.append(self.limit_order(symbol, best_bid, qty, "sell"))
                    position -= qty

        return orders
    
    def determine_tp_levels(self, magnitude: float) -> List[float]:
        if magnitude >= 0.01:
            return [0.24, 0.27, 0.3]
        elif magnitude >= 0.005:
            return [0.1, 0.2, 0.26]
        elif magnitude >= 0.0025:
            return [0.07, 0.12, 0.15]
        else:
            return [0.03, 0.05, 0.07]

    def run_macaron_strategy(self, state: TradingState) -> Dict[str, List[Order]]:

        result: Dict[str, List[Order]] = {}
        orders: List[Order] = []
        self.conversions = 0

        product = "MAGNIFICENT_MACARONS"
        if product not in state.order_depths or product not in state.observations.conversionObservations:
            return {}

        order_depth: OrderDepth = state.order_depths[product]
        obs = state.observations.conversionObservations[product]
        current_pos = state.position.get(product, 0)

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return {}

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        
        # === Update sunlight index history ===
        self.sunlight_indices.append(obs.sunlightIndex)

        if not obs:
            return {}
        if not obs.sunlightIndex:
            return {}

        if len(self.sunlight_indices) < self.sun_window:
            return {}

        smooth_sun = np.convolve(self.sunlight_indices, np.ones(self.sun_window) / self.sun_window, mode='valid')
        if len(smooth_sun) < 2:
            return {}

        gradient = np.gradient(smooth_sun)[-1]
        self.take_profit_levels = self.determine_tp_levels(abs(gradient))
        #print(f"[Tick {state.timestamp}] Gradient: {gradient:.6f} | Pos: {current_pos}")

        if abs(gradient) <= 0.005:
            return {}

        # === Detect gradient sign flip ===
        signal = None
        if self.previous_gradient is None:
            if gradient < 0:
                signal = 'long'
            elif gradient > 0:
                signal = 'short'
        else:
            if gradient < 0 and self.previous_gradient >= 0:
                signal = 'long'
            elif gradient > 0 and self.previous_gradient <= 0:
                signal = 'short'
        self.previous_gradient = gradient

        # === Entry logic on flip ===
        if signal and signal != self.last_position_side:
            self.entry_tick = state.timestamp
            self.position_side = signal
            self.entry_price = best_ask if signal == 'long' else best_bid
            self.take_profit_levels = self.determine_tp_levels(abs(gradient))
            self.profits_taken = [0] * len(self.take_profit_levels)
            self.last_position_side = signal
            #print(f"New {signal.upper()} entry @ {self.entry_price} | TPs: {self.take_profit_levels}")

        # === TP & Stop-Loss Logic ===
        desired_position = 0
        if self.entry_price and self.position_side:
            gain = (best_bid - self.entry_price) / self.entry_price if self.position_side == 'long' else (self.entry_price - best_ask) / self.entry_price
            if self.position_side == 'long' and self.entry_tick is not None:
                gain -= (0.001 * (state.timestamp - self.entry_tick)) / self.entry_price # Accounts for carry costs
            #print(f"Gain: {gain:.5f}")

            # === Stop Loss Check ===
            if gain <= self.stop_loss_threshold:
                #print(f"STOP LOSS triggered. Exiting full {self.position_side} position.")
                volume = min(abs(current_pos), best_bid_vol if current_pos > 0 else best_ask_vol)
                price = best_bid if current_pos > 0 else best_ask
                if volume > 0:
                    orders.append(Order(product, price, -volume if current_pos > 0 else volume))
                self.position_side = None
                self.entry_price = None
                self.entry_tick = None
                result[product] = orders
                return result

            # === Take Profit Logic ===
            tier_volume = self.max_position // len(self.take_profit_levels)
            remaining_pos = current_pos

            for i, level in enumerate(self.take_profit_levels):
                if gain >= level:
                    tp_vol = min(tier_volume - self.profits_taken[i], abs(remaining_pos))
                    if tp_vol > 0:
                        trade_vol = -tp_vol if self.position_side == 'long' else tp_vol
                        price = best_bid if self.position_side == 'long' else best_ask
                        orders.append(Order(product, price, trade_vol))
                        self.profits_taken[i] += tp_vol
                        remaining_pos -= trade_vol
                        #print(f"TP Hit ({level*100:.1f}%) | Vol: {tp_vol} @ {price}")

            # === Position Adjustment ===
            total_taken = sum(self.profits_taken)
            if self.position_side == 'long':
                desired_position = self.max_position - total_taken
            if self.position_side == 'short':
                desired_position = -self.max_position + total_taken
            #print(f"Desired Position: {desired_position}")

        # === Reset when fully exited ===
        if current_pos == 0 and self.position_side and self.entry_tick != state.timestamp:
            #print(f"Fully exited {self.position_side} position.")
            self.position_side = None
            self.entry_price = None
            self.entry_tick = None

        # Force flat position at tick 999400
        if 999400 <= state.timestamp <= 1000000:
            desired_position = 0

        # === Smooth adjustment ===
        diff = desired_position - current_pos
        if diff != 0:
            price = best_ask if diff > 0 else best_bid
            volume = min(abs(diff), best_ask_vol if diff > 0 else best_bid_vol, self.max_position_per_tick)
            if volume > 0:
                orders.append(Order(product, price, volume if diff > 0 else -volume))
                #print(f"Adjusting {'up' if diff > 0 else 'down'} {volume} @ {price}")
        
        if self.position_side == 'short':
            # === ARBITRAGE STRATEGY ===
            local_ask = obs.askPrice
            transport = obs.transportFees
            import_tariff = obs.importTariff
            local_ask_converted = local_ask + transport + import_tariff

            if best_bid > local_ask_converted:
                arb_volume = min(self.arb_max_size, best_bid_vol)
                self.conversions += arb_volume
                orders.append(Order(product, best_bid, -arb_volume))
                #print(f"ARBITRAGE: Buy Local @ {local_ask_converted:.2f} → Sell Foreign @ {best_bid} | Vol {arb_volume}")
            
        result[product] = orders
        return result
