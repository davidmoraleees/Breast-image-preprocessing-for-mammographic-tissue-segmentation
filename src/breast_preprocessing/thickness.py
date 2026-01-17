import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def find_farthest_point_from_chest_wall(skinline, image_width):
    """
    Return the contour point farthest from the chest
    wall (usually near the nipple).
    """
    chest_wall_x = image_width - 1
    distances = chest_wall_x - skinline[:, 1]
    farthest_point_index = np.argmax(distances)
    farthest_point = skinline[farthest_point_index]
    return farthest_point, farthest_point_index

def find_nearest_top(skinline):
    """Return the contour point closest to the top edge of the image."""
    top_point_index = np.argmin(skinline[:, 0])
    return skinline[top_point_index]

def find_nearest_right(skinline):
    """Return the contour point closest to the right edge of the image."""
    right_point_index = np.argmax(skinline[:, 1])
    return skinline[right_point_index]

def calculate_slope(point1, point2):
    """
    Compute the slope of the line joining two points,
    returning None for vertical lines.
    """
    if point1[1] == point2[1]:
        return None
    return (point2[0] - point1[0]) / (point2[1] - point1[1])

def find_intersection(skinline, slope, intercept):
    """
    Find the first contour point approximately intersecting
    a line defined by slope and intercept.
    """
    if slope is None:
        return None
    if np.isclose(slope, 0.0, atol=1e-8):
        for x, y in skinline:
            if np.isclose(intercept, x, atol=1.0):
                return [x, y]
        return None
    else:
        for x, y in skinline:
            if np.isclose(x, slope * y + intercept, atol=1.0):
                return [x, y]
        return None

def draw_reference_and_parallel_lines(skinline, offset_distance, num_lines, thickest_point):
    """
    Generate parallel contour-intersecting lines and
    select the one closest to the thickest point.
    """
    top_reference = find_nearest_top(skinline)
    right_reference = find_nearest_right(skinline)
    slope = calculate_slope(top_reference, right_reference)

    if slope is None:
        return [], None, slope

    intercept = top_reference[0] - slope * top_reference[1]

    parallel_lines = []
    min_distance = float("inf")
    closest_line = None

    for i in tqdm(range(num_lines), desc="Parallel lines"):
        parallel_intercept = intercept - (i + 1) * offset_distance / np.cos(np.arctan(slope))

        parallel_top = find_intersection(skinline, slope, parallel_intercept)
        parallel_bottom = find_intersection(skinline[::-1], slope, parallel_intercept)

        if parallel_top is not None and parallel_bottom is not None:
            parallel_lines.append((parallel_top, parallel_bottom))

            distance = np.abs(slope * thickest_point[1] - thickest_point[0] + parallel_intercept) / np.sqrt(slope**2 + 1)

            if distance < min_distance:
                min_distance = distance
                closest_line = ([float(parallel_top[0]), float(parallel_top[1])],
                                [float(parallel_bottom[0]), float(parallel_bottom[1])])

    return parallel_lines, closest_line, slope


def calculate_length(line):
    """Compute the Euclidean length of a line segment defined by two endpoints."""
    return np.sqrt((line[1][1] - line[0][1])**2 + (line[1][0] - line[0][0])**2)

def calculate_length_ratios(parallel_lines, reference_line):
    """Compute length ratios between each parallel line and a reference line."""
    reference_length = calculate_length(reference_line)
    if reference_length == 0:
        reference_length = 1e-5

    ratios = []
    for line in parallel_lines:
        line_length = calculate_length(line)
        ratios.append(line_length / reference_length)

    return ratios

def apply_length_ratios(image, skinline, ratios):
    """
    Apply thickness ratios across the image using
    distance from the skinline as interpolation index.
    """
    skinline_mask = np.zeros_like(image, dtype=np.uint8)
    for x, y in skinline:
        skinline_mask[int(x), int(y)] = 1

    distance_map = distance_transform_edt(skinline_mask)
    ratios_propagated = image.copy()
    max_distance = distance_map.max()

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            distance = distance_map[x, y]
            if distance > 0 and max_distance != 0:
                ratio_index = int((distance / max_distance) * (len(ratios) - 1))
                ratios_propagated[x, y] *= ratios[ratio_index]

    return ratios_propagated
