import asyncio
import threading
import websockets

connected_clients = set()
broadcast_bool = False

def set_broadcast_bool():
    global broadcast_bool
    broadcast_bool = True

async def broadcast(data):
    if connected_clients:
        message = data  # Ensure data is a string or serialize it
        await asyncio.gather(*[client.send(message) for client in connected_clients])
    await asyncio.sleep(1)  # Prevent spamming the message

async def websocket_handler(websocket, path):
    # Register client
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            # Handle incoming messages if needed
            pass
    finally:
        # Unregister client
        connected_clients.remove(websocket)

async def function_2():
    global broadcast_bool
    while True:
        if broadcast_bool:
            await broadcast("RE")
            broadcast_bool = False
        await asyncio.sleep(0.1)  # Prevent tight loop

async def main():
    server = await websockets.serve(websocket_handler, "localhost", 5055)
    broadcaster = asyncio.create_task(function_2())
    await asyncio.Future()  # Run forever

def input_loop():
    while True:
        if input("Enter 'broadcast' to send message to all clients: ") == "b":
            set_broadcast_bool()

if __name__ == "__main__":
    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()
    asyncio.run(main())
