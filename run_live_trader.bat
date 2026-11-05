@echo off
title Gotti Live News Trader Service
echo =========================================================================
echo  Starting Gotti Live News Trader Service (24/7 Lumibot + Alpaca Daemon)
echo  Strategies: News_LiquiditySweep + MarketReactor (Fade Mode)
echo =========================================================================

cd /d "%~dp0"
call ".\venv\Scripts\activate.bat"

python services\live_news_trader_service.py %*
pause
