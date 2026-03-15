from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import pymysql
import ssl
import sqlite3

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "service": "gotti-api"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Example endpoint to list strategies
@app.get("/strategies")
def list_strategies():
    strategies_dir = "strategies"
    strategies = []
    if os.path.exists(strategies_dir):
        for f in os.listdir(strategies_dir):
            if f.endswith(".py") and f != "__init__.py":
                strategies.append(f)
    return {"strategies": strategies}

def get_db_connection():
    try:
        # Provide path to the CA cert correctly
        ca_path = os.path.abspath(os.getenv("TIDB_CA_PATH", "isrgrootx1.pem"))
        
        ssl_ctx = ssl.create_default_context(cafile=ca_path)
        
        connection = pymysql.connect(
            host=os.getenv("TIDB_HOST"),
            port=int(os.getenv("TIDB_PORT", 4000)),
            user=os.getenv("TIDB_USER"),
            password=os.getenv("TIDB_PASSWORD"),
            database=os.getenv("TIDB_DATABASE"),
            ssl=ssl_ctx,
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

@app.get("/tidb-data")
def get_tidb_data():
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        with conn.cursor() as cursor:
            # Fetch some data. Since we don't know the schema, we'll return list of tables first
            # and attempt to fetch 10 rows from the first table if there's any.
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            
            data = {"tables": tables}
            
            if tables:
                first_table = list(tables[0].values())[0]
                cursor.execute(f"SELECT * FROM `{first_table}` LIMIT 10;")
                data["sample_data"] = cursor.fetchall()
                data["sample_table"] = first_table
            
            return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- Local Cloud Data Endpoints ---

def get_sqlite_connection():
    db_path = "local_cloud_data.db"
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # To return dict-like rows
    return conn

@app.get("/local-data/tables")
def get_local_tables():
    conn = get_sqlite_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Local database not found. Please run the load script first.")
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        return {"status": "success", "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/local-data/table/{table_name}")
def get_local_table_data(table_name: str, limit: int = 100, offset: int = 0):
    conn = get_sqlite_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Local database not found.")
    
    try:
        cursor = conn.cursor()
        
        # Verify table exists to prevent SQL injection
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")
            
        # Fetch data
        cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ? OFFSET ?;", (limit, offset))
        rows = [dict(row) for row in cursor.fetchall()]
        
        # Fetch total count
        cursor.execute(f"SELECT COUNT(*) as count FROM `{table_name}`;")
        total_count = cursor.fetchone()["count"]
        
        return {
            "status": "success", 
            "table": table_name,
            "total_rows": total_count,
            "data": rows,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
