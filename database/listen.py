import asyncio
import websockets

async def listen():
    uri = "ws://localhost:5055"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server")
        try:
            async for message in websocket:
                print(f"Received message: {message}")
        except websockets.ConnectionClosed:
            print("Connection closed")

if __name__ == "__main__":
    asyncio.run(listen())