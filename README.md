# Welcome to  Gotti - Visualizer

This repository is to be used alongside gotti-backend and gotti-frontend repositiories. 

This is the core repository that includes all trading strategies, backtesting, signal generation and data processing. 

![Visualization Example](AAPL_2024-09-21_15-57-59.png)

### To use run the following commands

```python
pip install --requirements.txt
streamlit run ./main.py
```

### Overview

The project is quite simple and has the following file structure

```none
gotti-visualize/
├── data/
├── indicators/
│   ├── data_processing.py
│   ├── indicator_calculations.py
│   └── visualization.py
├── tests/
│   ├── test_data_processing.py
│   ├── test_indicator_calculations.py
│   └── test_visualization.py
├── main.py
└── requirements.txt
```

### Setup

TA-LIB has been added to the project!! This means that numpy and pandas had to be downgraded as well!!
To install :

1. Create conda env.

```bash
conda create --name gotti python=3.12
conda activate 
```

 2. Install TALIB.

```python
conda install -c conda-forge libta-lib
conda install -c conda-forge ta-lib
```

3. Install requirements.

```python
cd path/to/project
pip install -r requirements.txt
```

### Common errors and workarounds

1. **Numpy and Pandas version conflict**
Downgrade numpy and pandas respectively

    ```python
    pip install "pandas<2.2.1" --force-reinstall 
    pip install "numpy<2.0.0" --force-reinstall 
    ```
2. **TA-LIB library missing**
For all TA-LIB related problems refer here: https://github.com/ta-lib/ta-lib-python
Or here : https://github.com/TA-Lib/ta-lib

### IMPORTANT

This repo might be used in the future **ONLY** to generate, preproccess and visualize data.
A new repo will be created to implement further data processing and machine learning models.
This is to ensure that the project remains clean, easy to understand and most important, beacuase numpy and pandas had to be downgraded to install TA-LIB.

### Visualisation

An oversimplified UI has been added to preview all sectors and all tickers registered under NASDAQ.
Volume and Market Cap filters have been applied to reduce cluttering and non-tradable (by my decision) symbols.To view simply run

```bash
cd .\data\
python app.py
```

And then proceed to

```bash
http://localhost:port/dashboard/
```

NOTE : Replace localhost:port with given parameters

### Contribution

We welcome contributions to this project! To contribute, please follow these steps:

1. **Fork the repository** to your own GitHub account.
2. **Clone the forked repository** to your local machine.
3. **Create a new branch** for your feature or bug fix:

    ```bash
    git checkout -b feature-name
    ```

4. **Make your changes** and commit them with clear and concise messages.
5. **Push your changes** to your forked repository:

    ```bash
    git push origin feature-name
    ```

6. **Create a pull request** from your branch to the main repository.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
