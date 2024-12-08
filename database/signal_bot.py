import asyncio
from telegram import Bot
import websockets

async def listen():
    uri = "ws://localhost:5055"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket server")
        try:
            async for message in websocket:
                bot = Bot(token="7924089058:AAHfnR2vcgBq3LRyKVKu4XdqfRu0ofQMI40")
                await bot.send_message(chat_id=8139983484, text=f"New Order: {message}")
                print(f"Received message: {message}")
        except websockets.ConnectionClosed:
            print("Connection closed")

if __name__ == "__main__":
    asyncio.run(listen())