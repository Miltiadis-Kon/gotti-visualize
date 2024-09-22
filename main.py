from indicators.support_resistance import get_ticker, find_support_resistance_levels, plot_support_resistance

def main():
    df_daily,major_support, major_resistance = find_support_resistance_levels('AAPL', '1d', '15m')
    plot_support_resistance(df_daily, major_support, major_resistance)
    

if __name__ == '__main__':
    main()