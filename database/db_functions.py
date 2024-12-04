import mysql.connector

def connect():
    return mysql.connector.connect(
        host="localhost",
        user = "root",
        password = "root",
        database = "mydatabase"
    )
 
 
'''
CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    side VARCHAR(50) NOT NULL,
    order_state VARCHAR(50) NOT NULL,
    stop_loss_price FLOAT,
    take_profit_price FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
);
'''

def insert_order(strategy, symbol, quantity, price, side, order_state, stop_loss_price, take_profit_price):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orders (strategy, symbol, quantity, price, side, order_state, stop_loss_price, take_profit_price) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (strategy, symbol, quantity, price, side, order_state, stop_loss_price, take_profit_price))
    conn.commit()
    cursor.close()
    conn.close()
    
def update_order(order_id, order_state):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE orders SET order_state = %s WHERE order_id = %s", (order_state, order_id))
    conn.commit()
    cursor.close()
    conn.close()
    
def delete_order(order_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM orders WHERE order_id = %s", (order_id,))
    conn.commit()
    cursor.close()
    conn.close()

def delete_orders(strategy=None, symbol=None, side=None, from_date=None, to_date=None):
    conn = connect()
    cursor = conn.cursor()
    query = "DELETE FROM orders WHERE 1=1"
    params = []
    
    if strategy:
        query += " AND strategy = %s"
        params.append(strategy)
    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)
    if side:
        query += " AND side = %s"
        params.append(side)
    if from_date:
        query += " AND created_at >= %s"
        params.append(from_date)
    if to_date:
        query += " AND created_at <= %s"
        params.append(to_date)
    
    cursor.execute(query, tuple(params))
    conn.commit()
    cursor.close()
    conn.close()
    
def get_orders(strategy=None, symbol=None, side=None, from_date=None, to_date=None):
        """
        Get orders based on various filters.
        
        Parameters:
        strategy (str): The strategy of the orders.
        symbol (str): The symbol of the orders.
        side (str): The side of the orders (e.g., 'long' or 'short').
        from_date (str): The start date for filtering orders.
        to_date (str): The end date for filtering orders.
        
        Returns:
        list: A list of orders matching the filters.
        
        Example:
        orders = get_orders_by_filter(strategy='strategy1', from_date='2023-01-01', to_date='2023-01-31')
        """
        conn = connect()
        cursor = conn.cursor()
        query = "SELECT * FROM orders WHERE 1=1"
        params = []
        
        if strategy:
            query += " AND strategy = %s"
            params.append(strategy)
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
        if side:
            query += " AND side = %s"
            params.append(side)
        if from_date:
            query += " AND created_at >= %s"
            params.append(from_date)
        if to_date:
            query += " AND created_at <= %s"
            params.append(to_date)
        
        cursor.execute(query, tuple(params))
        orders = cursor.fetchall()
        cursor.close()
        conn.close()
        return orders
 
'''
CREATE TABLE positions (
    position_id INT AUTO_INCREMENT PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    side VARCHAR(50) NOT NULL,
    stop_loss_price FLOAT,
    take_profit_price FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    based_on_order_id INT NOT NULL,
    FOREIGN KEY (based_on_order_id) REFERENCES orders(order_id),
    closed_at TIMESTAMP,
    closed_price FLOAT,
    closed_reason VARCHAR(50),
    profit_loss FLOAT
);
'''

def insert_position(strategy, symbol, quantity, price, side, stop_loss_price, take_profit_price, based_on_order_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO positions (strategy, symbol, quantity, price, side, stop_loss_price, take_profit_price, based_on_order_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (strategy, symbol, quantity, price, side, stop_loss_price, take_profit_price, based_on_order_id))
    conn.commit()
    cursor.close()
    conn.close()
    
def update_position(position_id, closed_price, closed_reason):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("UPDATE positions SET closed_price = %s, closed_reason = %s, closed_at = CURRENT_TIMESTAMP WHERE position_id = %s", (closed_price, closed_reason, position_id))
    conn.commit()
    cursor.close()
    conn.close()
    
def delete_position(position_id):
    conn = connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM positions WHERE position_id = %s", (position_id,))
    conn.commit()
    cursor.close()
    conn.close()
    
def delete_positions(strategy=None, symbol=None, side=None, from_date=None, to_date=None):
    conn = connect()
    cursor = conn.cursor()
    query = "DELETE FROM positions WHERE 1=1"
    params = []
    
    if strategy:
        query += " AND strategy = %s"
        params.append(strategy)
    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)
    if side:
        query += " AND side = %s"
        params.append(side)
    if from_date:
        query += " AND created_at >= %s"
        params.append(from_date)
    if to_date:
        query += " AND created_at <= %s"
        params.append(to_date)
    
    cursor.execute(query, tuple(params))
    conn.commit()
    cursor.close()
    conn.close()

def get_positions(strategy=None, symbol=None, side=None, from_date=None, to_date=None):
        """
        Get positions based on various filters.
        
        Parameters:
        strategy (str): The strategy of the positions.
        symbol (str): The symbol of the positions.
        side (str): The side of the positions (e.g., 'long' or 'short').
        from_date (str): The start date for filtering positions.
        to_date (str): The end date for filtering positions.
        
        Returns:
        list: A list of positions matching the filters.
        
        Example:
        positions = get_positions_by_filter(strategy='strategy1', from_date='2023-01-01', to_date='2023-01-31')
        """
        conn = connect()
        cursor = conn.cursor()
        query = "SELECT * FROM positions WHERE 1=1"
        params = []
        
        if strategy:
            query += " AND strategy = %s"
            params.append(strategy)
        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)
        if side:
            query += " AND side = %s"
            params.append(side)
        if from_date:
            query += " AND created_at >= %s"
            params.append(from_date)
        if to_date:
            query += " AND created_at <= %s"
            params.append(to_date)
        
        cursor.execute(query, tuple(params))
        positions = cursor.fetchall()
        cursor.close()
        conn.close()
        return positions