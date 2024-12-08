import pandas as pd
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def process_data(ml_data_path, uscities_path):
    data = pd.read_csv(ml_data_path)

    print("Loaded data")
    print("Shape: ", data.shape)

    # --- Data Cleaning and Preprocessing ---
    # Convert PickupDate to datetime and extract year and month
    data['PickupDate'] = pd.to_datetime(data['PickupDate'], errors='coerce')
    data['PickupYear'] = data['PickupDate'].dt.year
    data['PickupMonth'] = data['PickupDate'].dt.month

    # Remove rows with TotalRate not in the range [500, 2500]
    #data = data[(data['TotalRate'] >= 500) & (data['TotalRate'] <= 2500)]

    # --- Feature Engineering ---
    # Extract US state abbreviations from Pickup and Dropoff
    states = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    }

    def extract_state(text):
        if not isinstance(text, str):
            return None
        words = text.upper().split()
        for word in words:
            if word in states:
                return word
        return None

    data['Pickup State'] = data['Pickup'].apply(extract_state)
    data['Dropoff State'] = data['Dropoff'].apply(extract_state)

    # --- Load Cities Data ---
    cities = pd.read_csv(uscities_path)
    cities['city_only'] = cities['city_ascii'].str.upper()
    cities['state_only'] = cities['state_id'].str.upper()

    # Create lookup dictionaries for city information
    state_to_cities = cities.groupby('state_only')['city_only'].apply(set).to_dict()
    lookup = {
        (row['state_only'], row['city_only']): {
            'lat': row['lat'],
            'lng': row['lng'],
            'population': row['population']
        }
        for _, row in cities.iterrows()
    }

    # --- Assign Cities and Their Data ---
    for index, row in data.iterrows():
        try:
            pickup_state = row['Pickup State']
            dropoff_state = row['Dropoff State']

            # Process Pickup City
            if pickup_state in state_to_cities:
                possible_cities = state_to_cities[pickup_state]
                pickup_upper = row['Pickup'].upper()
                for city in possible_cities:
                    if city in pickup_upper:
                        data.at[index, 'Pickup City'] = city.title()
                        info = lookup.get((pickup_state, city))
                        if info:
                            data.at[index, 'Pickup Lat'] = info['lat']
                            data.at[index, 'Pickup Lng'] = info['lng']
                            data.at[index, 'Pickup Population'] = info['population']
                        break

            # Process Dropoff City
            if dropoff_state in state_to_cities:
                possible_cities = state_to_cities[dropoff_state]
                dropoff_upper = row['Dropoff'].upper()
                for city in possible_cities:
                    if city in dropoff_upper:
                        data.at[index, 'Dropoff City'] = city.title()
                        info = lookup.get((dropoff_state, city))
                        if info:
                            data.at[index, 'Dropoff Lat'] = info['lat']
                            data.at[index, 'Dropoff Lng'] = info['lng']
                            data.at[index, 'Dropoff Population'] = info['population']
                        break

        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Fill missing city names
    data['Pickup City'] = data['Pickup City'].fillna('Unknown')
    data['Dropoff City'] = data['Dropoff City'].fillna('Unknown')

    # Remove rows with unknown Pickup or Dropoff cities
    data = data[(data['Pickup City'] != 'Unknown') & (data['Dropoff City'] != 'Unknown')]

    # --- Calculate Haversine Distance AKA "as the crow flies" ---
    def haversine_distance(lat1, lng1, lat2, lng2):
        R = 3959  # Radius of Earth in miles
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    data['Distance (miles)'] = data.apply(
        lambda row: haversine_distance(row['Pickup Lat'], row['Pickup Lng'], row['Dropoff Lat'], row['Dropoff Lng']),
        axis=1
    )

    # -- Feature Engineering --
    # Encode categorical columns (states)
    state_encoder = LabelEncoder()
    data['Pickup State'] = state_encoder.fit_transform(data['Pickup State'])
    data['Dropoff State'] = state_encoder.fit_transform(data['Dropoff State'])

    # Create the 'Statetostate' feature
    data['Statetostate'] = data['Pickup State'].astype(str) + data['Dropoff State'].astype(str)
    statetostate_encoder = LabelEncoder()
    data['Statetostate'] = statetostate_encoder.fit_transform(data['Statetostate'])

    # --- Encode and Transform ---
    # Drop unused columns
    data = data.drop(columns=[
        'InvoiceNumber', 'LoadNumber', 'Pickup', 'Dropoff',
        'Pickup Lat', 'Pickup Lng', 'Dropoff Lat', 'Dropoff Lng',
        'Pickup City', 'Dropoff City', 'PickupDate'
    ])

    print("Final shape: ", data.shape)
    print("Features: ", data.columns)
    print("Data types: ", data.dtypes)

    # Create a correlation matrix
    corr_matrix = data.corr()
    # Plot the correlation matrix
    sns.heatmap(corr_matrix, annot=True)
    plt.title("Correlation Matrix of Features")
    plt.show()
    data = data[(data['TotalRate'] >= 500) & (data['TotalRate'] <= 2500)]
    # Drop Customer, PickupYear, PickupMonth, Pickup State, Dropoff State, Pickup Population since they are not useful
    data = data.drop(columns=['Customer', 'PickupYear', 'PickupMonth', 'Pickup State', 'Dropoff State', 'Pickup Population'])
    return data

