import os
import time
import pytz
import requests
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment


# Alpaca API credentials
ALPACA_API_KEY_ID = "PK201IEDCSCMOBTC17ML"
ALPACA_API_SECRET_KEY = "TOdqmGj3pQpdUwMTWRETIsNCj9DKhrTjhlbTDaeT"
BASE_URL = "https://paper-api.alpaca.markets/"


# Define target_price_series globally
STOCK_SYMBOL = ["AAPL"]
global_target_price = 1000
TARGET_PRICE = 1000

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, BASE_URL)



# Global definitions
USE_VOLUME = True  # Flag to enable or disable the volume condition
QUANTITY = 1
STOP_LOSS_CENTS = 0.20  # $0.20
TAKE_PROFIT_CENTS = 0.20  # $0.20
USE_ENGULFING = True  # Flag to enable or disable the engulfing strategy
VOLUME_FACTOR = 1  # Factor for high volume
USE_STRATEGY = False  # Flag to enable or disable the strategy
CHECK_PORTFOLIO = False

# Define the start date
START_DATE = datetime.now().strftime('%Y-%m-%d')  # Start date as today
END_DATE = datetime.now().strftime('%Y-%m-%d')  # End date as today

# Initialize the Alpaca client
api = tradeapi.REST(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, base_url=BASE_URL)
# Initialize the Alpaca client
client = StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)


async def get_bars_data(symbol, timeframe, start_date, end_date, adjustment='raw'):
    try:
        bars = await api.get_bars(symbol, 'minute', start_date, end_date, adjustment=adjustment, limit=50)
        return bars.df
    except Exception as e:
        print(f"Error occurred while retrieving bars data: {e}")
        return None


# Function to check for bullish engulfing pattern
def is_bullish_engulfing(df):
    if len(df) < 2:
        return False  # Not enough data for comparison

    last_index = -1
    prev_index = -2

    condition1 = df["close"].iloc[last_index] > df["open"].iloc[last_index]
    condition2 = df["open"].iloc[last_index] > df["close"].iloc[prev_index]
    condition3 = df["open"].iloc[prev_index] > df["close"].iloc[prev_index - 1]
    condition4 = df["close"].iloc[last_index] > df["open"].iloc[prev_index - 1]

    return all([condition1.any(), condition2.any(), condition3.any(), condition4.any()])


async def is_symbol_in_portfolio(api, symbol):
    try:
        if not CHECK_PORTFOLIO:  # Check if portfolio check is disabled
            print("Portfolio check is disabled.")
            return False
        positions = api.list_positions()
        if not positions:  # If positions list is empty
            print("No positions found.")
            return False
        my_positions_df = pd.DataFrame([position._raw for position in positions])
        if 'symbol' not in my_positions_df.columns:
            print("No positions found.")
            return False
        my_positions_df.set_index('symbol', inplace=True)
        if symbol in my_positions_df.index:
            print(f"{symbol} already has a position.")
            return True
        else:
            print(f"No positions found for {symbol}.")
            return False
    except Exception as e:
        print(f"Error occurred: {e}")
        return False






# Instantiate a data client
data_client = StockHistoricalDataClient(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY)


# Define the start and end time
start_time = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(minutes=45)

# Retrieve historical price data for each symbol
for symbol in STOCK_SYMBOL:
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_time
    )
    bars_df = data_client.get_stock_bars(request_params).df
    # Process bars_df for each symbol as needed

    # Assuming 'time' is the name of the column containing timestamps
    bars_df.index.name = 'timestamp'
    bars_df.reset_index(inplace=True)
    bars_df['timestamp'] = bars_df['timestamp'].dt.tz_convert('America/New_York')





import datetime
import pytz  # Import the pytz module for working with timezones

# Retrieve current time and make it timezone-aware
current_time = datetime.datetime.now(datetime.timezone.utc)
# Example timestamp from your data (replace this with your actual timestamp)
data_timestamp = datetime.datetime(2024, 5, 1, 12, 31, tzinfo=datetime.timezone.utc)

# Ensure both timestamps are timezone-aware
if not current_time.tzinfo:
    current_time = current_time.replace(tzinfo=datetime.timezone.utc)

# Compare timestamps
if current_time > data_timestamp:
    print("Current time is later than data timestamp.")
elif current_time < data_timestamp:
    print("Current time is earlier than data timestamp.")
else:
    print("Current time is equal to data timestamp.")

# Continue with the rest of your script...


















def is_high_volume(bars_df, symbol, volume_factor):
    # Calculate average volume for the specified symbol
    avg_volume = bars_df['volume'].mean()

    # Check if the last volume is greater than the average volume multiplied by the factor
    last_volume = bars_df['volume'].iloc[-1]
    return last_volume > avg_volume * volume_factor

# Place bracket order with stop-loss and take-profit
async def place_bracket_order(api, symbol, quantity, side, target_price_series, VOLUME_FACTOR):
    global global_target_price
    order_placed = set()  # Initialize a set to keep track of symbols for which orders have been placed
    for ticker in symbol:

        if await is_symbol_in_portfolio(api, ticker):
            print(f"{ticker} already has a position. Skipping order placement.")
            continue

        # Check if the order for this symbol has already been placed
        if ticker in order_placed:
            print(f"Order already placed for {ticker}. Skipping order placement.")
            continue
        else:
            order_placed.add(ticker)

        # Use the global target price for all symbols
        target_price = global_target_price

        # Retrieve latest trade data
        last_trade = api.get_latest_trade(ticker)
        current_price = last_trade.price

        # Compare current price with global target price
        if current_price > target_price:
            print(f"Current price for {ticker} is above the target price. Skipping order placement.")
            continue



        # Perform volume condition check
        if USE_VOLUME:
            avg_volume = bars_df['volume'].mean()
            last_volume = bars_df['volume'].iloc[-1]
            if last_volume <= avg_volume * VOLUME_FACTOR:
                print(f"Volume for {ticker} is not above average. Skipping order placement.")
                continue

        # Perform bullish engulfing pattern check
        if USE_ENGULFING:
            if not is_bullish_engulfing(bars_df):
                print(f"No bullish engulfing pattern found for {ticker}. Skipping order placement.")
                continue

        # Calculate stop loss and take profit prices in cents
        stop_loss_price = round(current_price - STOP_LOSS_CENTS, 2)
        take_profit_price = round(current_price + TAKE_PROFIT_CENTS, 2)

        # Ensure stop loss price is less than or equal to base price - 0.01
        stop_loss_price = max(stop_loss_price, 0)
        if stop_loss_price > target_price - 0.01:
            stop_loss_price = target_price - 0.01

        # Ensure take profit price is greater than or equal to base price + 0.01
        take_profit_price = max(take_profit_price, target_price + 0.01)

        # Submit bracket order
        bracket_order = api.submit_order(
            symbol=ticker,
            qty=quantity,
            side=side,
            type='limit',
            time_in_force='gtc',
            order_class='bracket',
            limit_price=target_price,
            stop_loss={'stop_price': stop_loss_price},
            take_profit={'limit_price': take_profit_price}
        )

        print(
            f"Buy order for {ticker} placed successfully with stop-loss at {stop_loss_price} cents and take-profit at {take_profit_price} cents.")

        import os
        from playsound import playsound

        # Play sound if order is placed successfully
        audio_file_name = "alert_sound.wav"  # Specify the name of your audio file
        audio_file_path = os.path.join(os.getcwd(), audio_file_name)  # Construct full path to audio file
        playsound(audio_file_path)  # Play the audio file
        print(f"Buy order for {ticker} placed successfully.")


# Main function
async def main():
    global CHECK_PORTFOLIO

    while True:
        # Initialize Alpaca API
        api = tradeapi.REST(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, BASE_URL)

        target_price_series = pd.Series([TARGET_PRICE] * len(STOCK_SYMBOL))
        order_placed = False  # Reset order_placed flag before each iteration

        # Fetch all open orders
        open_orders = api.list_orders(status='open')

        # Check if there are any inactive orders
        inactive_orders = [order for order in open_orders if order.status == 'inactive']
        print(f"There are {len(inactive_orders)} inactive orders.")
        await asyncio.sleep(1)  # Add a 1-second delay

        # Check if there are any working orders that have not been filled
        working_orders = [order for order in open_orders if order.status == 'working']
        unfilled_working_orders = [order for order in working_orders if order.filled_qty == 0]
        print(f"There are {len(unfilled_working_orders)} working orders that have not been filled.")
        await asyncio.sleep(1)  # Add a 1-second delay

        # Check if the symbol is already in the portfolio or has pending orders
        if CHECK_PORTFOLIO:  # Check if portfolio check is enabled
            for symbol in STOCK_SYMBOL:
                api = tradeapi.REST(ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, BASE_URL)
                positions = api.list_positions()

            if not positions:
                print(f"No positions found for {symbol}.")
            else:
                print(f"{symbol} already has a position.")
            await asyncio.sleep(1)  # Add a 1-second delay

        # Print the selected columns (this one has volume)
        print(bars_df.reset_index().loc[:, ['timestamp', 'open', 'close', 'high', 'low', 'volume']])

        # Place bracket order with stop-loss and take-profit
        await place_bracket_order(api, STOCK_SYMBOL, QUANTITY, 'buy', target_price_series, VOLUME_FACTOR)
        order_placed = True
        # Print the DataFrame with accurate timestamps



        # Place bracket order with stop-loss and take-profit
        await place_bracket_order(api, STOCK_SYMBOL, QUANTITY, 'buy', target_price_series, VOLUME_FACTOR)
        order_placed = True

        # Print the DataFrame with accurate timestamps
        bars_df_copy = bars_df.reset_index()  # Reset index to convert timestamp index to column
        bars_df_copy['timestamp'] = bars_df_copy['timestamp'].dt.tz_convert(
            'US/Eastern')  # Convert timestamps to Eastern Time (ET) timezone
        print(bars_df_copy[['timestamp', 'open', 'close', 'high', 'low', 'volume']])

        # Sleep for 5 seconds before starting the next iteration
        print("Before sleep")
        await asyncio.sleep(5)
        print("After sleep")

if __name__ == "__main__":
    asyncio.run(main())


