import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.morphology import medial_axis, skeletonize


def load_binary_mask(input_path):
    """Load black-on-white PNG as a foreground mask."""
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not loaded from {input_path}")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return img, binary


def preprocess_binary_mask(binary, kernel_size=3, iterations=1, approx_ratio=0.01):
    """Preprocess the binary mask to smooth contours and remove small jaggies."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=iterations)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(clean)
    for contour in contours:
        epsilon = max(1.0, approx_ratio * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.fillPoly(mask, [approx], 255)

    return mask


def clean_medial_axis_skeleton(binary):
    """Extract and prune a medial-axis skeleton using notebook-style spur removal."""
    skel_bool, _ = medial_axis(binary > 0, return_distance=True)
    dist_transform = distance_transform_edt(binary > 0)

    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    skel_conv = cv2.filter2D((skel_bool > 0).astype(np.uint8), -1, kernel)
    clean_skel = (skel_bool > 0).astype(np.uint8)

    for y, x in np.argwhere(skel_conv == 11):
        local_thickness = dist_transform[y, x]
        if local_thickness > 0:
            cv2.circle(clean_skel, (x, y), int(local_thickness * 0.5), 0, -1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_skel = cv2.morphologyEx(clean_skel, cv2.MORPH_CLOSE, kernel_close)
    return clean_skel > 0


def manual_thinning(binary_img):
    """Thin a binary stroke mask using a morphological thinning-like algorithm."""
    skel = np.zeros(binary_img.shape, dtype=np.uint8)
    img_copy = binary_img.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img_copy, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img_copy, temp)
        skel = cv2.bitwise_or(skel, temp)
        img_copy = eroded.copy()
        if cv2.countNonZero(img_copy) == 0:
            break
    return skel


def flux_based_centerline(binary_mask, flux_threshold=-0.4, min_distance=2):
    """Extract a centerline mask using flux-based ridge detection and thinning."""
    dist_transform = distance_transform_edt(binary_mask > 0)
    grad_x = cv2.Sobel(dist_transform, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dist_transform, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude[magnitude == 0] = 1.0
    nx = grad_x / magnitude
    ny = grad_y / magnitude

    flux_x = cv2.Sobel(nx, cv2.CV_64F, 1, 0, ksize=3)
    flux_y = cv2.Sobel(ny, cv2.CV_64F, 0, 1, ksize=3)
    flux = flux_x + flux_y

    centerline_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    centerline_mask[(flux < flux_threshold) & (dist_transform > min_distance)] = 255

    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    centerline_mask = cv2.morphologyEx(centerline_mask, cv2.MORPH_CLOSE, cleanup_kernel)
    centerline_mask = cv2.morphologyEx(centerline_mask, cv2.MORPH_OPEN, cleanup_kernel)

    skeleton_clean = manual_thinning(centerline_mask)
    return skeleton_clean > 0


def skeleton_pixels_to_graph(skel_bool):
    """
    Convert a skeleton image to an adjacency map.

    Diagonal shortcuts inside 2x2 blocks are ignored. That prevents false
    junctions where a one-pixel skeleton has both orthogonal and diagonal links.
    """
    h, w = skel_bool.shape
    pixels = set((int(x), int(y)) for y, x in np.argwhere(skel_bool))
    graph = {}

    for x, y in pixels:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if (nx, ny) not in pixels:
                    continue
                if dx != 0 and dy != 0:
                    if (x + dx, y) in pixels and (x, y + dy) in pixels:
                        continue
                neighbors.append((nx, ny))
        graph[(x, y)] = sorted(neighbors)

    return graph


def edge_key(a, b):
    return tuple(sorted((a, b)))


def trace_graph_branches(graph):
    """Trace graph edges between endpoints/junctions into ordered branches."""
    if not graph:
        return []

    special_nodes = {node for node, neighbors in graph.items() if len(neighbors) != 2}
    visited_edges = set()
    branches = []

    def trace_edge(start, first):
        branch = [start]
        previous = start
        current = first

        while True:
            visited_edges.add(edge_key(previous, current))
            branch.append(current)

            if current in special_nodes and current != start:
                break

            candidates = [
                n for n in graph[current]
                if n != previous and edge_key(current, n) not in visited_edges
            ]
            if not candidates:
                break

            incoming = np.array(current, dtype=float) - np.array(previous, dtype=float)
            current_arr = np.array(current, dtype=float)
            next_node = min(
                candidates,
                key=lambda p: np.linalg.norm((np.array(p, dtype=float) - current_arr) - incoming),
            )
            previous, current = current, next_node

        return branch

    for node in sorted(special_nodes):
        for neighbor in graph[node]:
            if edge_key(node, neighbor) not in visited_edges:
                branch = trace_edge(node, neighbor)
                if branch_length(branch) >= 4.0:
                    branches.append(branch)

    # Closed loops have no endpoint/junction. Trace any remaining edges.
    for node in sorted(graph):
        for neighbor in graph[node]:
            if edge_key(node, neighbor) not in visited_edges:
                branch = trace_edge(node, neighbor)
                if branch_length(branch) >= 4.0:
                    branches.append(branch)

    return branches


def branch_length(points):
    if len(points) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(np.array(points[i], dtype=float) - np.array(points[i - 1], dtype=float)))
        for i in range(1, len(points))
    )


def perpendicular_distances(points, start, end):
    pts = np.array(points, dtype=np.float32)
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)
    line = end - start
    length = float(np.linalg.norm(line))
    if length == 0:
        return np.linalg.norm(pts - start, axis=1)
    return np.abs(np.cross(line, start - pts)) / length


def project_to_fit_line(points):
    """Fit a line and project only the branch endpoints onto it."""
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    origin = np.array([x0, y0], dtype=np.float32)
    direction = np.array([vx, vy], dtype=np.float32)

    projected = []
    for point in (points[0], points[-1]):
        p = np.array(point, dtype=np.float32)
        t = float(np.dot(p - origin, direction))
        q = origin + t * direction
        projected.append((float(q[0]), float(q[1])))
    return projected


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker simplification for centerline cleanup."""
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]

    pts = np.array(points, dtype=np.float32)
    distances = perpendicular_distances(pts[1:-1], pts[0], pts[-1])
    if len(distances) == 0:
        return [(float(pts[0][0]), float(pts[0][1])), (float(pts[-1][0]), float(pts[-1][1]))]

    index = int(np.argmax(distances)) + 1
    max_distance = float(distances[index - 1])

    if max_distance > epsilon:
        left = rdp(points[: index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right

    return [(float(pts[0][0]), float(pts[0][1])), (float(pts[-1][0]), float(pts[-1][1]))]


def smooth_points(points, window=5):
    """Light moving-average smoothing that preserves branch endpoints."""
    if len(points) <= 2:
        return [(float(x), float(y)) for x, y in points]

    half = window // 2
    smoothed = []
    for i, point in enumerate(points):
        if i == 0 or i == len(points) - 1:
            smoothed.append((float(point[0]), float(point[1])))
            continue
        start = max(0, i - half)
        end = min(len(points), i + half + 1)
        section = np.array(points[start:end], dtype=np.float32)
        avg = np.mean(section, axis=0)
        smoothed.append((float(avg[0]), float(avg[1])))
    return smoothed


def angle_between(prev_point, point, next_point):
    """Return turn angle in degrees at point."""
    v1 = np.array(point, dtype=float) - np.array(prev_point, dtype=float)
    v2 = np.array(next_point, dtype=float) - np.array(point, dtype=float)
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 == 0 or len2 == 0:
        return 0.0
    cosine = float(np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def significant_turn_sum(points, threshold=12.0):
    """Return the amount of meaningful bend in a polyline."""
    if len(points) < 3:
        return 0.0

    total = 0.0
    for i in range(1, len(points) - 1):
        angle = turn_angle(points[i - 1], points[i], points[i + 1])
        if angle >= threshold:
            total += angle
    return total


def turn_sign_changes(points, threshold=8.0):
    """Count sign flips in local curvature, ignoring tiny jitter."""
    if len(points) < 4:
        return 0

    signs = []
    for i in range(1, len(points) - 1):
        a = np.array(points[i - 1], dtype=float)
        b = np.array(points[i], dtype=float)
        c = np.array(points[i + 1], dtype=float)
        v1 = b - a
        v2 = c - b
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        angle = turn_angle(a, b, c)
        if angle >= threshold and abs(cross) > 1e-6:
            signs.append(1 if cross > 0 else -1)

    if len(signs) < 2:
        return 0

    changes = 0
    prev = signs[0]
    for sign in signs[1:]:
        if sign != prev:
            changes += 1
            prev = sign
    return changes


def split_at_stroke_bends(points, angle_threshold=50.0, min_leg_length=20.0):
    """
    Split one traced skeleton branch into human stroke segments.

    Skeleton graph tracing only splits at topological junctions. Human strokes
    also split at strong geometric bends, such as the corner of a turn arrow.
    """
    if len(points) < 4:
        return [points]

    simplified = rdp(points, epsilon=3.0)
    if len(simplified) < 4:
        return [points]

    split_points = []
    cumulative = [0.0]
    for i in range(1, len(simplified)):
        cumulative.append(
            cumulative[-1]
            + float(np.linalg.norm(np.array(simplified[i]) - np.array(simplified[i - 1])))
        )
    total_length = cumulative[-1]

    for i in range(1, len(simplified) - 1):
        left_len = cumulative[i]
        right_len = total_length - cumulative[i]
        turn = angle_between(simplified[i - 1], simplified[i], simplified[i + 1])
        if turn >= angle_threshold and min(left_len, right_len) >= min_leg_length:
            split_points.append(simplified[i])

    if not split_points:
        return [points]

    split_indices = []
    for split_point in split_points:
        split_arr = np.array(split_point, dtype=float)
        nearest = min(
            range(1, len(points) - 1),
            key=lambda idx: np.linalg.norm(np.array(points[idx], dtype=float) - split_arr),
        )
        split_indices.append(nearest)

    split_indices = sorted(set(split_indices))
    segments = []
    start = 0
    for split_index in split_indices:
        segment = points[start: split_index + 1]
        if branch_length(segment) >= 4.0:
            segments.append(segment)
        start = split_index
    tail = points[start:]
    if branch_length(tail) >= 4.0:
        segments.append(tail)

    return segments if segments else [points]


def clean_centerline_branch(points):
    """
    Make one branch look more like a human stroke.

    If the branch is already close to a straight line, fit one clean line.
    Otherwise keep it curved, but smooth and simplify it gently.
    """
    if len(points) < 2:
        return []

    length = branch_length(points)
    if length < 4.0:
        return []

    simplified = rdp(points, epsilon=3.0)
    distances = perpendicular_distances(points, points[0], points[-1])
    straight_tolerance = max(4.0, length * 0.035)
    chord_length = float(np.linalg.norm(np.array(points[-1], dtype=float) - np.array(points[0], dtype=float)))
    max_distance = float(np.max(distances))

    straight_ratio = max_distance / chord_length if chord_length > 0 else 1.0

    straight_ratio_limit = 0.04 if chord_length >= 220.0 else 0.025
    if chord_length >= 80.0 and straight_ratio <= straight_ratio_limit:
        return project_to_fit_line(points)

    if len(simplified) <= 3:
        return project_to_fit_line(points)

    loose_straight_tolerance = max(8.0, length * 0.08)
    if chord_length < 220.0:
        loose_straight_tolerance = max(4.0, length * 0.025)
    if len(simplified) <= 5 and max_distance <= loose_straight_tolerance:
        return project_to_fit_line(points)

    if len(simplified) <= 4 and float(np.max(distances)) <= straight_tolerance:
        return project_to_fit_line(points)

    smoothed = smooth_points(points, window=7)
    return rdp(smoothed, epsilon=1.4)


def endpoint_line(path, endpoint_index):
    """Return a point and direction vector for the line at one path endpoint."""
    if endpoint_index == 0:
        point = np.array(path[0], dtype=float)
        neighbor = np.array(path[1], dtype=float)
    else:
        point = np.array(path[-1], dtype=float)
        neighbor = np.array(path[-2], dtype=float)

    direction = point - neighbor
    length = np.linalg.norm(direction)
    if length == 0:
        return point, None
    return point, direction / length


def fitted_line_meeting_point(cluster, paths):
    """
    Find the best shared point where endpoint tangents meet.

    This extends/shortens segments along their own directions instead of moving
    endpoints to an averaged blob point.
    """
    equations = []
    targets = []

    for path_index, endpoint_index, _ in cluster:
        point, direction = endpoint_line(paths[path_index], endpoint_index)
        if direction is None:
            continue
        normal_projection = np.eye(2) - np.outer(direction, direction)
        equations.append(normal_projection)
        targets.append(normal_projection @ point)

    points = np.array([endpoint[2] for endpoint in cluster], dtype=float)
    if len(equations) < 2:
        return tuple(np.mean(points, axis=0))

    a = np.vstack(equations)
    b = np.concatenate(targets)
    meeting, *_ = np.linalg.lstsq(a, b, rcond=None)
    meeting = np.array(meeting, dtype=float)

    if not np.all(np.isfinite(meeting)):
        return tuple(np.mean(points, axis=0))

    max_distance = float(np.max(np.linalg.norm(points - meeting, axis=1)))
    if max_distance > 160.0:
        return tuple(np.mean(points, axis=0))

    return (float(meeting[0]), float(meeting[1]))


def snap_close_endpoints(paths, radius=40.0):
    """
    Snap nearby segment endpoints to shared junction points.

    Stroke fitting can move branch endpoints slightly away from each other.
    Clustering only endpoints keeps the segment shape intact while making
    connected strokes render seamlessly. The shared point is computed from the
    endpoint tangent lines, so strokes lengthen to meet instead of being averaged.
    """
    endpoints = []
    for path_index, path in enumerate(paths):
        if len(path) < 2:
            continue
        endpoints.append((path_index, 0, np.array(path[0], dtype=float)))
        endpoints.append((path_index, -1, np.array(path[-1], dtype=float)))

    if len(endpoints) < 2:
        return paths

    parent = list(range(len(endpoints)))
    component_paths = [{endpoint[0]} for endpoint in endpoints]

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if component_paths[root_a] & component_paths[root_b]:
                return
            parent[root_b] = root_a
            component_paths[root_a].update(component_paths[root_b])

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if endpoints[i][0] == endpoints[j][0]:
                continue
            distance = float(np.linalg.norm(endpoints[i][2] - endpoints[j][2]))
            if distance <= radius:
                union(i, j)

    clusters = {}
    for index, endpoint in enumerate(endpoints):
        clusters.setdefault(find(index), []).append(endpoint)

    snapped_paths = [[tuple(point) for point in path] for path in paths]
    for cluster in clusters.values():
        if len(cluster) < 2:
            continue

        path_ids = [endpoint[0] for endpoint in cluster]
        if len(path_ids) != len(set(path_ids)):
            continue

        shared = fitted_line_meeting_point(cluster, snapped_paths)

        endpoint_map = {}
        for path_index, endpoint_index, _ in cluster:
            endpoint_map.setdefault(path_index, set()).add(endpoint_index)

        collapsed_paths = {
            path_index for path_index, endpoint_indices in endpoint_map.items()
            if len(endpoint_indices) == 2
        }

        for path_index, endpoint_index, _ in cluster:
            if path_index in collapsed_paths:
                continue
            snapped_paths[path_index][endpoint_index] = shared

    return snapped_paths


def endpoint_interior_point(path, endpoint_index, distance):
    """Point inside a path, measured from one endpoint along the path."""
    if len(path) < 2:
        return None

    remaining = distance
    current_index = 0 if endpoint_index == 0 else len(path) - 1
    step = 1 if endpoint_index == 0 else -1
    current = np.array(path[current_index], dtype=float)

    while 0 <= current_index + step < len(path):
        next_index = current_index + step
        next_point = np.array(path[next_index], dtype=float)
        segment = next_point - current
        segment_length = float(np.linalg.norm(segment))
        if segment_length == 0:
            current_index = next_index
            current = next_point
            continue
        if remaining <= segment_length:
            point = current + (segment / segment_length) * remaining
            return (float(point[0]), float(point[1]))
        remaining -= segment_length
        current_index = next_index
        current = next_point

    return tuple(path[current_index])


def set_endpoint(path, endpoint_index, point):
    updated = list(path)
    updated[endpoint_index] = point
    return updated


def nearest_skeleton_point(point, skeleton_points, radius=45.0):
    """Move a point back to the closest original skeleton pixel nearby."""
    if skeleton_points is None or len(skeleton_points) == 0:
        return point

    point_arr = np.array(point, dtype=float)
    distances = np.linalg.norm(skeleton_points - point_arr, axis=1)
    nearest_index = int(np.argmin(distances))
    if float(distances[nearest_index]) > radius:
        return point

    nearest = skeleton_points[nearest_index]
    return (float(nearest[0]), float(nearest[1]))


def round_connected_corners(paths, skel_bool=None, endpoint_radius=2.0, corner_radius=34.0):
    """
    Replace hard two-stroke endpoint intersections with a small curved bridge.

    The bridge is created only from local geometry: two endpoints must already
    meet, and the turn angle must be a real corner instead of a straight merge.
    """
    endpoints = []
    for path_index, path in enumerate(paths):
        if len(path) < 2:
            continue
        endpoints.append((path_index, 0, np.array(path[0], dtype=float)))
        endpoints.append((path_index, -1, np.array(path[-1], dtype=float)))

    rounded_paths = [list(path) for path in paths]
    connectors = []
    skeleton_points = None
    if skel_bool is not None:
        ys, xs = np.nonzero(skel_bool)
        skeleton_points = np.column_stack((xs.astype(float), ys.astype(float)))

    parent = list(range(len(endpoints)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if endpoints[i][0] == endpoints[j][0]:
                continue
            distance = float(np.linalg.norm(endpoints[i][2] - endpoints[j][2]))
            if distance <= endpoint_radius:
                union(i, j)

    clusters = {}
    for index, endpoint in enumerate(endpoints):
        clusters.setdefault(find(index), []).append(endpoint)

    for cluster in clusters.values():
        if len(cluster) != 2:
            continue
        (path_a, endpoint_a, point_a), (path_b, endpoint_b, point_b) = cluster
        if path_a == path_b:
            continue

        cluster = [(path_a, endpoint_a, point_a), (path_b, endpoint_b, point_b)]
        shared = fitted_line_meeting_point(cluster, rounded_paths)
        centered_shared = nearest_skeleton_point(shared, skeleton_points, radius=0.5)
        trim = min(
            corner_radius,
            branch_length(rounded_paths[path_a]) * 0.28,
            branch_length(rounded_paths[path_b]) * 0.28,
        )
        if trim < 8.0:
            continue

        inner_a = endpoint_interior_point(rounded_paths[path_a], endpoint_a, trim)
        inner_b = endpoint_interior_point(rounded_paths[path_b], endpoint_b, trim)
        if inner_a is None or inner_b is None:
            continue

        corner_angle = turn_angle(inner_a, centered_shared, inner_b)
        if not 45.0 <= corner_angle <= 140.0:
            continue

        rounded_paths[path_a] = set_endpoint(rounded_paths[path_a], endpoint_a, inner_a)
        rounded_paths[path_b] = set_endpoint(rounded_paths[path_b], endpoint_b, inner_b)
        connectors.append([inner_a, centered_shared, inner_b])

    return rounded_paths + connectors


def quadratic_bridge_points(start, control, end, samples=9):
    """Sample a smooth quadratic bridge."""
    points = []
    for i in range(samples):
        t = i / (samples - 1)
        mt = 1.0 - t
        x = mt * mt * start[0] + 2.0 * mt * t * control[0] + t * t * end[0]
        y = mt * mt * start[1] + 2.0 * mt * t * control[1] + t * t * end[1]
        points.append((float(x), float(y)))
    return points


def smooth_three_way_junctions(paths, endpoint_radius=2.0, angle_min=70.0, angle_max=145.0):
    """
    Keep a three-way junction anchored on the original centerline.

    We avoid inventing a new bridge here so the source stroke can naturally
    cover the intersection the way a hand-drawn `6` does.
    """
    endpoints = []
    for path_index, path in enumerate(paths):
        if len(path) < 2:
            continue
        endpoints.append((path_index, 0, np.array(path[0], dtype=float)))
        endpoints.append((path_index, -1, np.array(path[-1], dtype=float)))

    if len(endpoints) < 3:
        return paths

    parent = list(range(len(endpoints)))
    component_paths = [{endpoint[0]} for endpoint in endpoints]

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if component_paths[root_a] & component_paths[root_b]:
                return
            parent[root_b] = root_a
            component_paths[root_a].update(component_paths[root_b])

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if endpoints[i][0] == endpoints[j][0]:
                continue
            distance = float(np.linalg.norm(endpoints[i][2] - endpoints[j][2]))
            if distance <= endpoint_radius:
                union(i, j)

    clusters = {}
    for index, endpoint in enumerate(endpoints):
        clusters.setdefault(find(index), []).append(endpoint)

    rounded_paths = [list(path) for path in paths]

    for cluster in clusters.values():
        if len(cluster) != 3:
            continue

        path_ids = [endpoint[0] for endpoint in cluster]
        if len(path_ids) != len(set(path_ids)):
            continue

        directions = []
        for path_index, endpoint_index, _ in cluster:
            _, direction = endpoint_line(rounded_paths[path_index], endpoint_index)
            if direction is None:
                break
            directions.append((path_index, endpoint_index, direction))
        if len(directions) != 3:
            continue

        pair_candidates = []
        for i in range(3):
            for j in range(i + 1, 3):
                dir_a = directions[i][2]
                dir_b = directions[j][2]
                angle = float(np.degrees(np.arccos(np.clip(np.dot(dir_a, dir_b), -1.0, 1.0))))
                pair_candidates.append((angle, i, j))

        angle, i, j = min(pair_candidates, key=lambda item: item[0])
        if not (angle_min <= angle <= angle_max):
            continue

        path_i = rounded_paths[directions[i][0]]
        path_j = rounded_paths[directions[j][0]]
        bend_i = significant_turn_sum(path_i)
        bend_j = significant_turn_sum(path_j)
        if max(bend_i, bend_j) < 20.0:
            continue

        shared = np.mean([endpoint[2] for endpoint in cluster], axis=0)
        shared_point = nearest_skeleton_point(tuple(shared), None)
        for path_index, endpoint_index, _ in cluster:
            rounded_paths[path_index] = set_endpoint(
                rounded_paths[path_index],
                endpoint_index,
                shared_point,
            )

    return rounded_paths


def prune_redundant_endpoint_pairs(paths, straight_turn_limit=12.0, curved_turn_limit=20.0, length_ratio=0.75):
    """
    Drop a short, near-straight duplicate when a curved stroke already connects
    the same two junctions.

    For looped shapes like `6`, skeleton tracing can produce both the real loop
    and a chord across the join. Keeping the loop makes the rebuilt stroke flow
    into the circle more naturally.
    """
    groups = {}
    for path_index, path in enumerate(paths):
        if len(path) < 2:
            continue
        start = (round(path[0][0], 2), round(path[0][1], 2))
        end = (round(path[-1][0], 2), round(path[-1][1], 2))
        key = tuple(sorted((start, end)))
        groups.setdefault(key, []).append((path_index, path))

    remove = set()
    for group in groups.values():
        if len(group) < 2:
            continue

        stats = []
        for path_index, path in group:
            chord = float(np.linalg.norm(np.array(path[-1], dtype=float) - np.array(path[0], dtype=float)))
            if chord <= 1e-6:
                continue
            distances = perpendicular_distances(path, path[0], path[-1])
            deviation = float(np.max(distances)) / chord if len(distances) else 0.0
            stats.append(
                {
                    "index": path_index,
                    "length": branch_length(path),
                    "turn": significant_turn_sum(path),
                    "deviation": deviation,
                }
            )

        if len(stats) < 2:
            continue

        curved = [item for item in stats if item["turn"] >= curved_turn_limit or item["deviation"] >= 0.05]
        straight = [item for item in stats if item["turn"] <= straight_turn_limit and item["deviation"] <= 0.03]
        if not curved or not straight:
            continue

        best_curved = max(curved, key=lambda item: (item["turn"], item["length"]))
        for item in straight:
            if item["length"] <= best_curved["length"] * length_ratio:
                remove.add(item["index"])

    return [path for index, path in enumerate(paths) if index not in remove]


def merge_smooth_endpoint_pairs(paths, endpoint_radius=2.0, max_angle=35.0):
    """
    Merge two strokes when they meet as a smooth continuation.

    Unlike `merge_aligned_segments`, this allows longer curved pieces. It only
    works at simple two-path junctions, so real three-way intersections stay
    split as separate strokes.
    """
    merged_paths = [list(path) for path in paths]

    changed = True
    while changed:
        changed = False

        endpoints = []
        for path_index, path in enumerate(merged_paths):
            if len(path) < 2:
                continue
            endpoints.append((path_index, 0, np.array(path[0], dtype=float)))
            endpoints.append((path_index, -1, np.array(path[-1], dtype=float)))

        candidates = []
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                path_a, endpoint_a, point_a = endpoints[i]
                path_b, endpoint_b, point_b = endpoints[j]
                if path_a == path_b:
                    continue
                if float(np.linalg.norm(point_a - point_b)) > endpoint_radius:
                    continue

                incident = 0
                shared = (point_a + point_b) / 2.0
                for _, _, point in endpoints:
                    if float(np.linalg.norm(point - shared)) <= endpoint_radius:
                        incident += 1
                if incident != 2:
                    continue

                oriented_a = list(reversed(merged_paths[path_a])) if endpoint_a == 0 else list(merged_paths[path_a])
                oriented_b = list(merged_paths[path_b]) if endpoint_b == 0 else list(reversed(merged_paths[path_b]))
                if len(oriented_a) < 2 or len(oriented_b) < 2:
                    continue

                angle = turn_angle(oriented_a[-2], tuple(shared), oriented_b[1])
                if angle <= max_angle:
                    combined = oriented_a[:-1] + [(float(shared[0]), float(shared[1]))] + oriented_b[1:]
                    candidates.append((angle, path_a, path_b, combined))

        if not candidates:
            continue

        _, path_a, path_b, combined = min(candidates, key=lambda item: item[0])
        keep = min(path_a, path_b)
        drop = max(path_a, path_b)
        merged_paths[keep] = combined
        del merged_paths[drop]
        changed = True

    return merged_paths


def turn_angle(a, b, c):
    """Angle between segment a->b and b->c."""
    v1 = np.array(b, dtype=float) - np.array(a, dtype=float)
    v2 = np.array(c, dtype=float) - np.array(b, dtype=float)
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 == 0 or len2 == 0:
        return 180.0
    cosine = float(np.clip(np.dot(v1, v2) / (len1 * len2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def best_merge_candidate(path_a, path_b, max_angle=22.0, endpoint_radius=2.0):
    """Return merged path when two paths meet and continue in almost the same direction."""
    if len(path_a) > 4 or len(path_b) > 4:
        return None

    candidates = []
    orientations_a = [path_a, list(reversed(path_a))]
    orientations_b = [path_b, list(reversed(path_b))]

    for oriented_a in orientations_a:
        for oriented_b in orientations_b:
            joint_distance = float(
                np.linalg.norm(np.array(oriented_a[-1], dtype=float) - np.array(oriented_b[0], dtype=float))
            )
            if joint_distance > endpoint_radius:
                continue
            if len(oriented_a) < 2 or len(oriented_b) < 2:
                continue

            angle = turn_angle(oriented_a[-2], oriented_a[-1], oriented_b[1])
            if angle <= max_angle:
                shared = (
                    (oriented_a[-1][0] + oriented_b[0][0]) / 2.0,
                    (oriented_a[-1][1] + oriented_b[0][1]) / 2.0,
                )
                merged = oriented_a[:-1] + [shared] + oriented_b[1:]
                candidates.append((angle, merged))

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[0])[1]


def merge_aligned_segments(paths, max_angle=22.0):
    """
    Merge false splits where two stroke pieces meet and continue straight.

    This fixes cases like K where a small junction connector and the actual
    diagonal belong to the same human stroke, and vertical halves should become
    one vertical stroke.
    """
    merged_paths = [list(path) for path in paths]

    changed = True
    while changed:
        changed = False
        for i in range(len(merged_paths)):
            if changed:
                break
            for j in range(i + 1, len(merged_paths)):
                merged = best_merge_candidate(merged_paths[i], merged_paths[j], max_angle=max_angle)
                if merged is None:
                    continue
                merged_paths[i] = merged
                del merged_paths[j]
                changed = True
                break

    return merged_paths


def dedupe_paths(paths, precision=2):
    seen = set()
    unique_paths = []
    for path in paths:
        if len(path) < 2:
            continue
        key = tuple((round(point[0], precision), round(point[1], precision)) for point in path)
        rev = tuple(reversed(key))
        if key in seen or rev in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def simplify_final_stroke(path, deviation_ratio=0.05, short_deviation_ratio=0.015, long_chord=220.0):
    """Collapse a final stroke to one line when it is still nearly straight."""
    if len(path) < 3:
        return path
    if has_sharp_corner(path):
        return path

    if significant_turn_sum(path) > 55.0:
        return path

    chord = float(np.linalg.norm(np.array(path[-1], dtype=float) - np.array(path[0], dtype=float)))
    if chord < 1.0:
        return path

    distances = perpendicular_distances(path, path[0], path[-1])
    deviation = float(np.max(distances)) / chord
    if deviation <= short_deviation_ratio or (chord >= long_chord and deviation <= deviation_ratio):
        return [path[0], path[-1]]

    return path


def simplify_final_strokes(paths):
    """Clean exported strokes after snapping/merging."""
    return [simplify_final_stroke(path) for path in paths]


def path_to_svg_d(points):
    if not points:
        return ""
    if len(points) == 3:
        d = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
        d += f" L {points[1][0]:.2f} {points[1][1]:.2f}"
        d += f" L {points[2][0]:.2f} {points[2][1]:.2f}"
        return d
    if len(points) >= 8 and not has_sharp_corner(points):
        return path_to_quadratic_svg_d(points)

    d = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
    for point in points[1:]:
        d += f" L {point[0]:.2f} {point[1]:.2f}"
    return d


def has_sharp_corner(points, threshold=50.0):
    """Detect polygonal strokes that should stay as straight-line commands."""
    for i in range(1, len(points) - 1):
        if turn_angle(points[i - 1], points[i], points[i + 1]) >= threshold:
            return True
    return False


def path_to_quadratic_svg_d(points):
    """Use smooth quadratic curves for long curved strokes."""
    d = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
    for i in range(1, len(points) - 1):
        control = points[i]
        end = (
            (points[i][0] + points[i + 1][0]) / 2.0,
            (points[i][1] + points[i + 1][1]) / 2.0,
        )
        d += f" Q {control[0]:.2f} {control[1]:.2f} {end[0]:.2f} {end[1]:.2f}"

    control = points[-2]
    end = points[-1]
    d += f" Q {control[0]:.2f} {control[1]:.2f} {end[0]:.2f} {end[1]:.2f}"
    return d


def dedupe_paths(paths, precision=2):
    seen = set()
    unique_paths = []
    for path in paths:
        if len(path) < 2:
            continue
        key = tuple((round(point[0], precision), round(point[1], precision)) for point in path)
        rev = tuple(reversed(key))
        if key in seen or rev in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def export_to_svg(paths, filename, width=1024, height=1024, group_id="skeleton-shapes", preserve_aspect_ratio="none"):
    path_elements = []
    for i, path in enumerate(paths, 1):
        if len(path) < 2:
            continue
        path_elements.append(
            f'<path id="path-{i}" d="{path_to_svg_d(path)}" '
            'fill="none" stroke="black" stroke-width="45" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    svg_content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="{preserve_aspect_ratio}">\n'
        f'<g id="{group_id}">\n' + "\n".join(path_elements) + "\n</g>\n</svg>\n"
    )
    with open(filename, "w") as f:
        f.write(svg_content)


def draw_overlay(base_img, paths):
    overlay = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    for path in paths:
        pts = np.array(path, dtype=np.int32)
        for i in range(len(pts) - 1):
            cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 3)
    return overlay


def process_geometric_centerline(input_path, output_prefix="debug", debug=False, mode="medial"):
    img, binary = load_binary_mask(input_path)
    if debug:
        cv2.imwrite(f"{output_prefix}_01_binary.png", binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        contour_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_vis, contours, -1, (255, 0, 0), 2)
        cv2.imwrite(f"{output_prefix}_02_contour.png", contour_vis)

    if mode == "flux":
        skel_bool = flux_based_centerline(binary)
    else:
        skel_bool = clean_medial_axis_skeleton(binary)

    skeleton = (skel_bool.astype(np.uint8) * 255)
    if debug:
        cv2.imwrite(f"{output_prefix}_03_skeleton.png", skeleton)

    graph = skeleton_pixels_to_graph(skel_bool)
    raw_branches = trace_graph_branches(graph)
    fixed_paths = []
    for branch in raw_branches:
        for segment in split_at_stroke_bends(branch, angle_threshold=30.0, min_leg_length=8.0):
            cleaned = clean_centerline_branch(segment)
            if len(cleaned) >= 2 and branch_length(cleaned) >= 40.0:
                fixed_paths.append(cleaned)

    fixed_paths = snap_close_endpoints(fixed_paths)
    fixed_paths = merge_aligned_segments(fixed_paths)
    fixed_paths = snap_close_endpoints(fixed_paths)
    fixed_paths = round_connected_corners(fixed_paths, skel_bool=skel_bool, endpoint_radius=2.0, corner_radius=34.0)
    fixed_paths = smooth_three_way_junctions(fixed_paths)
    fixed_paths = prune_redundant_endpoint_pairs(fixed_paths)
    fixed_paths = merge_smooth_endpoint_pairs(fixed_paths)
    fixed_paths = [path for path in fixed_paths if branch_length(path) >= 20.0]
    fixed_paths = simplify_final_strokes(fixed_paths)
    fixed_paths = dedupe_paths(fixed_paths)

    if debug:
        overlay = draw_overlay(img, fixed_paths)
        cv2.imwrite(f"{output_prefix}_04_fixed_overlay.png", overlay)
    export_to_svg(fixed_paths, f"{output_prefix}_fixed.svg", width=img.shape[1], height=img.shape[0])

    print(f"Raw branches: {len(raw_branches)}")
    print(f"Fixed paths: {len(fixed_paths)}")
    if debug:
        print(f"Wrote {output_prefix}_04_fixed_overlay.png and {output_prefix}_fixed.svg")
    else:
        print(f"Wrote {output_prefix}_fixed.svg")


def process_all_cases(sample_dir="challenge_sample", output_dir="challenge_sample_results"):
    """Run the fixed centerline workflow for every PNG sample."""
    sample_dir = Path(sample_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(sample_dir.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {sample_dir}")
        return

    print(f"Found {len(png_files)} PNG files")
    print(f"Output: {output_dir}")
    print()

    for png_file in png_files:
        output_prefix = output_dir / png_file.stem
        print(f"Processing {png_file.name}")
        process_geometric_centerline(png_file, str(output_prefix))
        print()
