CREATE DATABASE gotti;

USE gotti;

CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL,
    memebership_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    order_id VARCHAR(50) NOT NULL ,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT,
    side VARCHAR(50) NOT NULL,
    order_state VARCHAR(50) NOT NULL,
    stop_loss_price FLOAT,
    take_profit_price FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE positions (
    position_id VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    side VARCHAR(50) NOT NULL,
    stop_loss_price FLOAT,
    take_profit_price FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    based_on_order_id VARCHAR(50) NOT NULL,
    position_state VARCHAR(50) NOT NULL, 
    closed_at TIMESTAMP,
    closed_price FLOAT,
    closed_reason VARCHAR(50),
    profit_loss FLOAT
);