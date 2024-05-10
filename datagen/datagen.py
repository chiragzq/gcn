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
import cv2
from stitching import Stitcher



load_dotenv()

GAPI_KEY = os.getenv("GAPI_KEY")

gmaps = googlemaps.Client(key=GAPI_KEY)

stitcher = Stitcher(confidence_threshold=0.2)

def sample_lat_long():
    return math.asin(random.uniform(-1,1)) * 180 / math.pi, random.uniform(-180, 180)

def request_metadata(lat, lng, radius):
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?source=outdoor&location={lat},{lng}&radius={radius}&key={GAPI_KEY}"
    return requests.get(url).json()

def request_pano(pano_id, heading):
    url = f"https://maps.googleapis.com/maps/api/streetview?fov=120&size=640x640&pano={pano_id}&heading={heading}&key={GAPI_KEY}"
    return requests.get(url).content

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
        radius = 1000
        lat, lng = sample_lat_long()
        metadata = request_metadata(lat, lng, radius)
        if metadata["status"] == "OK":
            return metadata["pano_id"], metadata["location"]["lat"], metadata["location"]["lng"]

def generate_points(n=10, input_file=None,output_file="dataset/coords_country.csv"):
    if input_file:
        df = pd.read_csv(input_file)
    else:
        df = pd.DataFrame(columns=["pano_id", "latitude", "longitude", "country"])
    for i in range(n):
        pano_id, lat, lng = sample_point()
        country = lat_long_to_country(lat, lng)
        print(country)
        if country:
            df = pd.concat([df, pd.DataFrame([[pano_id, lat, lng, country]],columns=["pano_id", "latitude", "longitude", "country"])])
        df.to_csv(output_file, index=False)

def download_pano(input_csv="dataset/coords_country.csv", output_dir="dataset/img", start_index=794):
    data = pd.read_csv(input_csv)
    for i, row in data.iterrows():
        if i < start_index:
            continue
        pano_id = row["pano_id"]
        # images = []
        for heading in [0,120]:
            img = request_pano(pano_id,heading)
            with open(f"{output_dir}/{i}_{heading}.png", "wb") as f:
                f.write(img)
            print(f"Downloaded {pano_id}_{heading}.png")
            cv2_img = cv2.imread(f"{output_dir}/{i}_{heading}.png")
            cv2_img = cv2_img[:590]
            cv2.imwrite(f"{output_dir}/{i}_{heading}.png", cv2_img)
            # images.append(cv2_img)
            # os.remove(f"{output_dir}/{i}_{heading}.png")
        # panorama = stitcher.stitch(images)

        # cv2.imwrite(f"{output_dir}/{i}.png", panorama) 

def plot_points(input_file="dataset/coords_country.csv"):
    data = pd.read_csv(input_file)
    geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
    gdf = GeoDataFrame(data, geometry=geometry)   
    # print(gdf)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=5)
    plt.show()

def find_empty_countries(input_file="dataset/coords_country.csv"):
    data = pd.read_csv(input_file)
    empty_countries = data[data["country"].isnull()]
    print(empty_countries)

def print_countries(input_file="dataset/coords_country.csv"):
    data = pd.read_csv(input_file)
    print(data["country"].unique())

def process_img_low(in_dir="dataset/img", out_dir="dataset/img_low"):
    n = len(os.listdir(in_dir)) // 2
    for i in range(n):
        img1 = cv2.imread(f"{in_dir}/{i}_0.png")
        img2 = cv2.imread(f"{in_dir}/{i}_120.png")
        img = cv2.hconcat([img1, img2])
        # half res
        img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        cv2.imwrite(f"{out_dir}/{i}.png", img)


if __name__ == "__main__":
    # plot_points()
    # find_empty_countries()
    # print_countries()
    # print(sample_point())
    # generate_points(n=10000, input_file="dataset/coords_country.csv")
    # download_pano()
    process_img_low()

#https://github.com/RuurdBijlsma/random-streetview/blob/master/src/StreetView.js#L260