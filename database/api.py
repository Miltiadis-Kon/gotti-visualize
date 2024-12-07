from flask import Flask, request, jsonify
import db_functions as sql

app = Flask(__name__)

@app.route('/orders', methods=['POST'])
def create_order():
    data = request.json
    sql.insert_order(
        strategy=data['strategy'],
        symbol=data['symbol'],
        quantity=data['quantity'],
        price=data['price'],
        side=data['side'],
        order_id=data['order_id'],
        order_state=data['order_state'],
        stop_loss_price=data.get('stop_loss_price'),
        take_profit_price=data.get('take_profit_price')
    )
    #TODO: send order to telegram bot
    set_last_order(data) # Update the last order in the cache
    return jsonify({"message": "Order created successfully"}), 201


last_order=None # Cache the last order

def set_last_order(order): # Set the last order in the cache
    global last_order
    last_order=order



@app.route('/last_order', methods=['GET']) # Get the last order
def get_last_order():
    return jsonify(last_order), 200
    

@app.route('/orders', methods=['GET'])
def get_orders():
    strategy = request.args.get('strategy')
    symbol = request.args.get('symbol')
    side = request.args.get('side')
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    state = request.args.get('order_state')
    orders = sql.get_orders(strategy, symbol, side,state,from_date, to_date)
    return jsonify(orders), 200

@app.route('/orders/<int:order_id>', methods=['PUT'])
def update_order(order_id):
    data = request.json
    sql.update_order(order_id, data['order_state'])
    return jsonify({"message": "Order updated successfully"}), 200

@app.route('/orders/<int:order_id>', methods=['DELETE'])
def delete_order(order_id):
    sql.delete_order(order_id)
    return jsonify({"message": "Order deleted successfully"}), 200

@app.route('/positions', methods=['POST'])
def create_position():
    data = request.json
    sql.insert_position(
        strategy=data['strategy'],
        symbol=data['symbol'],
        quantity=data['quantity'],
        price=data['price'],
        side=data['side'],
        stop_loss_price=data.get('stop_loss_price'),
        take_profit_price=data.get('take_profit_price'),
        based_on_order_id=data['based_on_order_id'],
        position_state=data['position_state']
    )
    sql.update_order(data['based_on_order_id'], 'FILLED')
    return jsonify({"message": "Position created successfully"}), 201

@app.route('/positions', methods=['GET'])
def get_positions():
    strategy = request.args.get('strategy')
    symbol = request.args.get('symbol')
    side = request.args.get('side')
    from_date = request.args.get('from_date')
    to_date = request.args.get('to_date')
    positions = sql.get_positions(strategy, symbol, side, from_date, to_date)
    return jsonify(positions), 200

@app.route('/positions/<int:position_id>', methods=['PUT'])
def update_position(position_id):
    data = request.json
    sql.update_position(position_id, data['closed_price'], data['closed_reason'], data['position_state'])
    return jsonify({"message": "Position updated successfully"}), 200

@app.route('/positions/<int:position_id>', methods=['DELETE'])
def delete_position(position_id):
    sql.delete_position(position_id)
    return jsonify({"message": "Position deleted successfully"}), 200

if __name__ == '__main__':
    app.run(debug=True)