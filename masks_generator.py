import geojson
from shapely.geometry import shape
from PIL import Image, ImageDraw
from xml.dom import minidom


GEOJSON_PATH = 'raw_data/building-polygon.geojson'
PATH = 'raw_data/Yaroslavl_planet_order_296167/'
DIRS = {'20180824_080111_1044': (8584, 2772), '20180824_080112_1044': (8595, 2761),
        '20180824_080113_1044': (8585, 2771), '20180824_080114_1044': (8572, 2757)}
BUILDING_TYPES = {}
BUILDING_LEVELS = {}


def coord_to_pix(coord):
    lng, lat = coord
    x = w / (max_lat - min_lat) * (lat - min_lat)
    y = h / (max_lng - min_lng) * (lng - min_lng)
    return (x, y)


def draw_polygon(image, coordinates):
    d = ImageDraw.Draw(image)
    d.polygon(coordinates, fill=(255, 255, 255))
    return image


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
        bt = str(el['properties']['BUILDING']).replace(' ', '')
        bl = str(el['properties']['B_LEVELS']).replace(' ', '')
        local_polygon = shape(el['geometry'])
        if polygon.contains(local_polygon):
            polygons.append(el['geometry']['coordinates'][0])
            if bt not in BUILDING_TYPES.keys():
                BUILDING_TYPES[bt] = 1
            else:
                BUILDING_TYPES[bt] += 1
            if bl not in BUILDING_LEVELS.keys():
                BUILDING_LEVELS[bl] = 1
            else:
                BUILDING_LEVELS[bl] += 1
    return polygons


def output_statistics():
    print(80 * '-')
    print('Building types:')
    for k in sorted(BUILDING_TYPES.keys()):
        print('{:25} {}'.format(k, BUILDING_TYPES[k]))
    print(80 * '-')
    print('Building levels:')
    for k in sorted(BUILDING_LEVELS.keys()):
        print('{:25} {}'.format(k, BUILDING_LEVELS[k]))


with open(GEOJSON_PATH, encoding='utf-8') as f:
    data = geojson.load(f)
print('Features:\t\t\t {}'.format(len(data['features'])))
stage = 1
for dir_name in DIRS.keys():
    print(80*'-')
    print('Directory:\t\t\t', dir_name)
    metadata_path = PATH + dir_name + '/' + dir_name + '_3B_AnalyticMS_metadata.xml'
    metadata = minidom.parse(metadata_path)
    w, h = 8448, 2560
    max_lat = max([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:latitude')])
    max_lng = max([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:longitude')])
    min_lat = min([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:latitude')])
    min_lng = min([float(x.firstChild.data) for x in metadata.getElementsByTagName('ps:longitude')])
    file_name = 'data_transformed/' + dir_name + '.jpg'
    size = (w, h)
    img = Image.new('RGB', size, (0, 0, 0))
    print('Geo coordinates:\t', max_lat, max_lng, min_lat, min_lng)
    polygons = select_polygons(max_lat, max_lng, min_lat, min_lng)
    print('Polygons:\t\t\t', len(polygons))
    for polygon in polygons:
        coordinates = []
        for x in polygon:
            if len(x) == 2:
                coordinates.append(coord_to_pix(x))
        if len(coordinates) != 0:
            image = draw_polygon(img, coordinates)
    img.convert("RGB").save(file_name, subsampling=0, quality=100)
    print('Generated masks:\t {}/{}'.format(stage, len(DIRS)))
    stage += 1
output_statistics()