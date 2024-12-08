from flask import Flask
import asyncio
import websockets
import threading
from queue import Queue
import json

app = Flask(__name__)
clients = set()
message_queue = Queue()

async def handler(websocket):
    clients.add(websocket)
    try:
        while True:
            await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def broadcast_messages():
    async with websockets.serve(handler, "localhost", 5055):
        while True:
            if not message_queue.empty():
                message = message_queue.get()
                if clients:
                    websockets_tasks = [
                        client.send(json.dumps(message))
                        for client in clients
                    ]
                    await asyncio.gather(*websockets_tasks)
            await asyncio.sleep(0.1)

def run_websocket_server():
    asyncio.run(broadcast_messages())

# Start WebSocket server in a separate thread
websocket_thread = threading.Thread(target=run_websocket_server)
websocket_thread.daemon = True
websocket_thread.start()

@app.route('/send/<message>')
def send_message(message):
    message_queue.put({"data": message})
    return f"Message '{message}' queued for broadcast"

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)