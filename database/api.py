from flask import Flask, request, jsonify
import db_functions as sql
import asyncio
import websockets
import threading
from queue import Queue
import json

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
    message_queue.put({"data": data})
    return jsonify({"message": "Order created successfully"}), 201


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

clients = set() # Store all connected clients
message_queue = Queue() # Queue to store messages to be sent to clients

async def handler(websocket): # Handler for WebSocket connections
    clients.add(websocket)
    try:
        while True:
            await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def broadcast_messages():   # Broadcast messages to all connected clients
    async with websockets.serve(handler, "localhost", 5055):
        while True:
            if not message_queue.empty(): # If there are messages in the queue
                message = message_queue.get() # Get the message
                if clients:
                    websockets_tasks = [
                        client.send(json.dumps(message)) # Send the message to all connected clients
                        for client in clients
                    ]
                    await asyncio.gather(*websockets_tasks) # Wait for all messages to be sent
            await asyncio.sleep(0.1) 

def run_websocket_server(): # Run the WebSocket server
    asyncio.run(broadcast_messages()) # Run the broadcast_messages function in an event loop

# Start WebSocket server in a separate thread
websocket_thread = threading.Thread(target=run_websocket_server) # Create a thread to run the WebSocket server
websocket_thread.daemon = True # Set the thread as a daemon so it will be terminated when the main thread exits
websocket_thread.start() # Start the thread



if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)