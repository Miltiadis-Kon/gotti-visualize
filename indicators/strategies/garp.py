'''
https://www.simplertrading.com/products/free-courses/mastering-the-trade?view=ch_7__part_2_opening_gap

Table 7.1 
Table 7.2

Garp method is heavily inspired by trader's sentimental feedback on news.
News will influence a stock and that provides the gap.
Then reaction to the news will either indicate a fill of the gap or a trend continuation.

A gap occurs when the opening price of the next day’s 
regular cash session is greater or lower than the closing price of the previous day’s 
regular cash session, creating a “gap” in price levels on the charts,


As Zed once said ALL gaps will eventually be filled ! 

Gaps provide a very sucessfull strategy and are LAAAAZY ! Perfect to automate.

" premarket volume can tell a trader if the gap is going to be a professional breakaway event
or is going to lead to price action that has a high probability 
of filling the gap on the very same day it was created.
"

Remember, this is a fade play.
I will buy a gap down and short a gap up.
'''



# Lets keep track of each individual indictor 

# 1. Pre market volume 
def get_premarket_volume(ticker):
    """
    Get the premarket volume of a ticker relative to contract size
    """
    # Get the volume of the VIX index to determine the overall volatility 
    vix = get_ticker("VIX")
    
    # Get contract size 
    contract_size = get_ticker(ticker).contract_size
    
    # Compare vix with contract size to adjust the contract size 
    adjusted_contract_size = adjust_contract_size(contract_size,vix)
        
    return adjusted_contract_size


def adjust_contract_size(contract_size,vix):
    '''
    If VIX ~ 20 => contract_size - No change:
    If VIX ~ 30 => contract_size * 1.5:
    If VIX ~ 40 => contract_size * 2:
    If VIX ~ 60 => contract_size * 3: 
    '''
    if vix >= 55:
        adjusted_size = contract_size * 3
    elif vix >= 45:
        adjusted_size = contract_size * 2.5
    elif vix >= 35:
        adjusted_size = contract_size * 2
    elif vix >= 25:
        adjusted_size = contract_size * 1.5
    else:
        adjusted_size = contract_size
    return adjusted_size



def entry_exit_strategy(adjusted_contract_size):
    """
    Pre market volume | Position size | Trade Target
    x < 30k           | Full size     | Exit at gap fill
    30k > x >70k      | 2/3 size      | Exit half at 50% & half at gap fill
    x > 70k           | No trade      | No exit
    """
    
    


# 2. Statistical gap size analysis relative to the trading day
def get_probability_of_gap_fill(ticker):
    """
    Get the probability of a gap fill based on historical data
    """
    pass

# 3. Gap size 
def get_gap_size(ticker):
    """
    Get the size of the gap
    """
    # Get last day closing price
    
    # Get today's  open price
    
    # Find the difference
    pass

# 4. Gap corelation with  overall market movement
def get_market_corelation(volume):
    """
    Get the corelation of the gap with the market index SNP,NASDAQ,DOW
    """
    # Basically if NAS,DOW,SNP has a similar % gap then we have no reason to trade
    
    pass

def garp(ticker):
    '''
    
    '''
    volume = get_premarket_volume(ticker)
    corelation = get_market_corelation(volume)
    if not corelation:
        return "No trade"
    probability = get_probability_of_gap_fill(ticker)
    size = get_gap_size(ticker)
    return "Trade"    