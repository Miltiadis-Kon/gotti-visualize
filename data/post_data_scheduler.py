import os
import json
from flask import Flask, request, jsonify
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
from io import StringIO

app = Flask(__name__)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Kaggle dataset details
DATASET_OWNER = 'your-kaggle-username'
DATASET_NAME = 'your-dataset-name'
DATABASE_FILE = 'your-database-file.csv'  # Assuming CSV for this example

@app.route('/update_database', methods=['POST'])
def update_database():
    try:
        # Get the update data from the request
        data = request.json
        row_id = data.get('id')
        updates = data.get('updates')

        # Download the current database file
        api.dataset_download_file(f'{DATASET_OWNER}/{DATASET_NAME}', DATABASE_FILE)

        # Read the CSV file
        df = pd.read_csv(DATABASE_FILE)

        # Update the specified row
        if row_id in df['id'].values:
            for column, value in updates.items():
                df.loc[df['id'] == row_id, column] = value
        else:
            return jsonify({'error': 'Row not found'}), 404

        # Save the updated DataFrame to a CSV file
        df.to_csv(DATABASE_FILE, index=False)

        # Create a new version of the dataset
        api.dataset_create_version(f'{DATASET_OWNER}/{DATASET_NAME}',
                                   {DATABASE_FILE: DATABASE_FILE},
                                   f'Updated row {row_id}',
                                   quiet=False)

        # Clean up: remove the local copy of the database file
        os.remove(DATABASE_FILE)

        return jsonify({'message': 'Database updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)