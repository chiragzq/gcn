from dotenv import load_dotenv
import pandas as pd
import os
import googlemaps
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import random
import requests
import math


load_dotenv()

GAPI_KEY = os.getenv("GAPI_KEY")
print(GAPI_KEY)

gmaps = googlemaps.Client(key=GAPI_KEY)

def sample_lat_long():
    return math.asin(random.uniform(-1,1)) * 180 / math.pi, random.uniform(-180, 180)

def request_metadata(lat, lng, radius):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lng}&radius={radius}&key={GAPI_KEY}"
    return requests.get(url).json()

def lat_long_to_country(latitude, longitude):
    try:
        reverse_geocode_result = gmaps.reverse_geocode((latitude, longitude), result_type="country")
        country = reverse_geocode_result[0]["address_components"][0]["long_name"]
        return country
    except Exception as e:
        print("Error converting ", (latitude, longitude), " to country", e)
        return None

def sample_point():
    while True:
        radius = 10000
        lat, lng = sample_lat_long()
        metadata = request_metadata(lat, lng, radius)
        if metadata["status"] == "OK":
            return metadata["pano_id"], metadata["location"]["lat"], metadata["location"]["lng"]


def convert_dataset(input_file="dataset/0coords.csv", output_file="dataset/0coords_country.csv"):
    data = pd.read_csv(input_file)
    for index, row in data.iterrows():
        country = lat_long_to_country(row["latitude"], row["longitude"])
        data.at[index, "country"] = country
        print("Converted ", (row["latitude"], row["longitude"]), " to ", country)
    data.to_csv(output_file, index=True)
    
def find_empty_countries(input_file="dataset/0coords_country.csv"):
    data = pd.read_csv(input_file)
    empty_countries = data[data["country"].isnull()]
    print(empty_countries)

def plot_points(input_file="dataset/0coords_country.csv"):
    data = pd.read_csv(input_file)
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    gdf = GeoDataFrame(data, geometry=geometry)   
    # print(gdf)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=5)
    plt.show()

if __name__ == "__main__":
    # convert_dataset()

    # find_empty_countries()

    # plot_points()

    print(sample_point())