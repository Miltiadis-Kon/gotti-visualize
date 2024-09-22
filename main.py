from indicators.support_resistance_window import find_support_resistance_levels, plot_support_resistance, plot_support_resistance_ploty
from indicators.support_resistance_pivotpoints import find_pivot_points, plot_support_resistance_pivotpoints


def compare(major_support,major_resistance,pivot_support,pivot_resistance,minor_threshold=0.1) :
        # Compare major support and resistance levels with pivot support and resistance levels
    major_support = sorted(major_support)
    major_resistance = sorted(major_resistance)
    pivot_support = sorted(pivot_support)
    pivot_resistance = sorted(pivot_resistance)
    # Compare major support and resistance levels with pivot support and resistance levels to find the common levels
    # Add a minor threshold for comparison to account for small differences in the levels due to the algorithm
    common_support = []
    common_resistance = []
    for s in major_support:
        for ps in pivot_support:
            if abs(s - ps) < minor_threshold:
                common_support.append(s)
    for r in major_resistance:
        for pr in pivot_resistance:
            if abs(r - pr) < minor_threshold:
                common_resistance.append(r)
    print_sr(major_support,major_resistance,pivot_support,pivot_resistance,common_support,common_resistance)   
    return common_support, common_resistance

def print_sr(major_support,major_resistance,pivot_support,pivot_resistance,common_support,common_resistance):
    print('*'*40)
    print('STRICTLY FOR DEBUGGING PURPOSES')        
    print('*'*40)        
    print('Major Support Levels:', major_support)
    print('Pivot Support Levels:', pivot_support)
    print('Common Support Levels:', common_support)
    print('*'*40)        
    print('Major Resistance Levels:', major_resistance)
    print('Pivot Resistance Levels:', pivot_resistance) 
    print('Common Resistance Levels:', common_resistance)
    print('*'*40) 

def main(plot=True,plot_common=False):
    #TODO: Define dictionary or algo to assign timeframe to frequency and window size ratio 
    # (1d,15m = 20window,2freq is ok) but for 5d it clutters a lot 
    data,major_support, major_resistance = find_support_resistance_levels('AAPL', '5d', '15m')
    data,pivot_support, pivot_resistance = find_pivot_points('AAPL', '5d', '15m',10,10)
    
    #TODO: Define a threshold for comparison of support and resistance levels
    minor_threshold = 0.1 # means 0.1$ difference is allowed between SR methods
    
    # Compare major support and resistance levels with pivot support and resistance levels 
    common_support,common_resistance = compare(major_support,major_resistance,pivot_support,pivot_resistance,minor_threshold)
    if plot_common:
        plot_support_resistance_ploty(data, common_support, common_resistance)
        
    
    if plot:
        #plot_support_resistance(df_daily, major_support, major_resistance)
        plot_support_resistance_ploty(data, major_support, major_resistance)
        plot_support_resistance_pivotpoints(data)

if __name__ == '__main__':
    main(plot=False,plot_common=True)