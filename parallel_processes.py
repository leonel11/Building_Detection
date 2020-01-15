import geojson
from shapely.geometry import shape, Point
from PIL import Image
import numpy as np
from xml.dom import minidom
import multiprocessing
import math


GEOJSON_PATH = 'raw_data/building-polygon.geojson'
PATH = 'raw_data/Yaroslavl_planet_order_296167/'
DIRS = ['20180824_080111_1044', '20180824_080112_1044', '20180824_080113_1044', '20180824_080114_1044']


def pix_to_coord(x, y):
    lat = (max_lat - min_lat) / w * x + min_lat
    lng = (max_lng - min_lng) / h * y + min_lng
    return (lng, lat)


def select_polygons(max_lat, max_lng, min_lat, min_lng):
    polygons = []
    geojson = {
        "type": "Polygon",
        "coordinates": [
            [
                [min_lng, min_lat],
                [max_lng, min_lat],
                [max_lng, max_lat],
                [min_lng, max_lat],
                [min_lng, min_lat]
            ]
        ]
    }
    polygon = shape(geojson)
    for el in data['features']:
        local_polygon = shape(el['geometry'])
        if polygon.contains(local_polygon):
            polygons.append(local_polygon)
    return polygons


def parallel_function(polygons, img, start_pos, end_pos, w, h, max_lat, max_lng, min_lat, min_lng):
    print(start_pos, end_pos)
    for x in range(0, w):
        for y in range(start_pos, end_pos):
            lat = (max_lat - min_lat) / w * x + min_lat
            lng = (max_lng - min_lng) / h * y + min_lng
            point = Point(lng, lat)
            include = False
            for polygon in polygons:
                if polygon.contains(point):
                    include = True
                    break
            if include:
                img[y, x] = 255


with open(GEOJSON_PATH, encoding='utf-8') as f:
    data = geojson.load(f)
stage = 0
for dir_name in DIRS:
    print('Extracted data:\t{}/{}'.format(stage+1, len(DIRS)))
    print(dir_name)
    metadata_path = PATH + dir_name + '/' + dir_name + '_3B_AnalyticMS_metadata.xml'
    metadata = minidom.parse(metadata_path)
    w = int(metadata.getElementsByTagName('ps:numColumns')[0].firstChild.data)
    h = int(metadata.getElementsByTagName('ps:numRows')[0].firstChild.data)
    print('Image size:\t{} x {}'.format(w, h))
    max_lat = max([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:latitude')])
    max_lng = max([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:longitude')])
    min_lat = min([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:latitude')])
    min_lng = min([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:longitude')])
    file_name = 'data' + '/' + dir_name + '.jpg'
    img = np.zeros((h, w), dtype=np.uint8)
    print(max_lat, max_lng, min_lat, min_lng)
    polygons = select_polygons(max_lat, max_lng, min_lat, min_lng)
    cpus = multiprocessing.cpu_count()
    print(cpus)
    borders = list(map(math.floor, list(np.arange(0, w, cpus+1))))
    procs = []
    for i in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=parallel_function, args=(polygons, img, borders[i], borders[i+1],
                                                                    w, h, max_lat, max_lng, min_lat, min_lng))
        procs.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    Image.fromarray(img).save(file_name)
    stage+=1