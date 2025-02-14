# Step 16: Load Data Using Yahoo Finance
@st.cache_data
def load_data(symbol="BTC-USD", interval="1d", period="5y"):
    try:
        st.write(f"Loading data for {symbol} with interval {interval} and period {period}")
        df = fetch_yahoo_data(symbol, interval, period)
        
        if df is None or df.empty:
            st.warning(f"⚠️ No data available for {symbol}.")
            return None

        # Calculate all indicators (comment them out and add one by one)
        df = calculate_bollinger_bands(df)
        st.write("After Bollinger Bands:", df.shape)

        df = calculate_macd(df)
        st.write("After MACD:", df.shape)

        df = calculate_rsi(df)
        st.write("After RSI:", df.shape)

        df = calculate_atr(df)
        st.write("After ATR:", df.shape)

        df = calculate_adx(df)
        st.write("After ADX:", df.shape)

        df = calculate_fibonacci_levels(df)
        st.write("After Fibonacci Levels:", df.shape)

        df = calculate_ichimoku(df)
        st.write("After Ichimoku Cloud:", df.shape)

        df = calculate_vwap(df)
        st.write("After VWAP:", df.shape)

        df = calculate_obv(df)
        st.write("After OBV:", df.shape)

        df = calculate_moving_averages(df)
        st.write("After Moving Averages:", df.shape)

        df = calculate_stochastic_oscillator(df)
        st.write("After Stochastic Oscillator:", df.shape)

        df.dropna(inplace=True)
        df = df.astype(np.float64)

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None
    return df
