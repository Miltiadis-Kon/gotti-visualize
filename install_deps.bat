@echo off
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo Upgrading pip and wheel...
python -m pip install --upgrade pip
pip install wheel

echo Installing dependencies...
echo Installing pandas (latest binary)...
pip install pandas

echo Installing pandas_ta (ignoring dependencies to avoid version conflicts)...
pip install pandas_ta --no-deps

echo Installing remaining requirements...
pip install -r requirements.txt
