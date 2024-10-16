'''
SOURCE CODE : https://demo.scichart.com/javascript/stock-chart-buy-sell-markers
'''

# app.py
from flask import Flask, render_template
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    # Load your stock data here
    # This is just example data
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=20),
        'open': [100 + i + np.random.randn() for i in range(100)],
        'high': [105 + i + np.random.randn() for i in range(100)],
        'low': [95 + i + np.random.randn() for i in range(100)],
        'close': [102 + i + np.random.randn() for i in range(100)],
    })
    
    # Convert date column to string to make it JSON serializable
    df['date'] = df['date'].astype(str)
    
    # Convert DataFrame to JSON
    chart_data = df.to_dict(orient='records')
    
    lengths = {key :len(value) for key, value in chart_data.items()}
    print(lengths)
    
    return render_template('index.html', chart_data=json.dumps(chart_data))

if __name__ == '__main__':
    app.run(debug=True)