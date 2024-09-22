# Welcome to		Gotti - Visualizer !

This repository will be used to **process and visualize** data and **technical indicators**  of a single stock ticker **using python**. The goal is to implement an automated UI **similar to TradingView**.

![Visualization Example](AAPL_2024-09-21_15-57-59.png)

### To use run the following commands:

```
pip install --requirements.txt
python -m main.py
```
### Overview
The project is quite simple and has the following file structure
```
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

### Contribution
We welcome contributions to this project! To contribute, please follow these steps:

1. **Fork the repository** to your own GitHub account.
2. **Clone the forked repository** to your local machine.
3. **Create a new branch** for your feature or bug fix:
    ```
    git checkout -b feature-name
    ```
4. **Make your changes** and commit them with clear and concise messages.
5. **Push your changes** to your forked repository:
    ```
    git push origin feature-name
    ```
6. **Create a pull request** from your branch to the main repository.

Please ensure your code adheres to the project's coding standards and includes appropriate tests.

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.