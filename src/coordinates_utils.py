import math
from geopy.distance import distance

def distance_to_segment(point, start, end):
    segment_distance = distance(start, end).m
    start_2_point_distance = distance(start, point).m
    end_2_point_distance = distance(end, point).m

    if segment_distance > start_2_point_distance and segment_distance > end_2_point_distance:
        # Théorème de la médiane
        mediane = math.sqrt((start_2_point_distance**2 + end_2_point_distance**2 - (segment_distance**2/2))/2)
        mediane_height_gap = abs(start_2_point_distance**2 - end_2_point_distance**2) / (2*segment_distance)
        height = math.sqrt(mediane**2 - mediane_height_gap**2)
        return height
    else :
        return min(start_2_point_distance, end_2_point_distance)

def is_point_close_segment(point, start, end, distance_threshold=50):
    distance = distance_to_segment(point, start, end)
    if distance < distance_threshold:
        return True
    else:
        return False

def format_coords(x):
    x_str = str(x)
    if x_str[0] == "-" and len(x_str) > 1:
        return -float(x_str[1:].replace(",", "."))
    elif x_str[0] != "-":
        return float(x_str.replace(",", "."))
    else:
        return -200