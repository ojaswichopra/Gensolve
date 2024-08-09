import numpy as np

from scipy.spatial.distance import euclidean
import matplotlib . pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from scipy.spatial import ConvexHull, QhullError

# Tolerance for considering endpoints as connected
ENDPOINT_TOLERANCE = 10
COUNT = 0

# Function to read CSV file and visualize 
def read_csv_o(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    # plt.show ()

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    
    # Initialize a list to hold all shapes
    shapes = []
    
    # Iterate over each unique shape identifier (first column)
    for shape_id in np.unique(np_path_XYs[:, 0]):
        shape = []
        
        # Get all paths for this shape
        npXYs = np_path_XYs[np_path_XYs[:, 0] == shape_id][:, 1:]
        
        # Iterate over each unique path identifier (second column)
        for path_id in np.unique(npXYs[:, 0]):
            # Get all points for this path and store them in a numpy array
            path = npXYs[npXYs[:, 0] == path_id][:, 1:]
            shape.append(path)
        
        # Append the shape (list of paths) to the shapes list
        shapes.append(shape)
    
    return shapes

# Function to regularize all paths in a shape
def regularize_shape(shape):
    return [regularize_path(path) for path in shape]
    
# Function to regularize a path to a straight line or arc of a circle
def regularize_path(path):
    # if len(path) < 2:
    #     return path
    globals()["COUNT"] += 1
    # combined_points = np.vstack(path)
    is_line, line_points = fit_line(path)
    if is_line:
            return line_points     
    else:     
        print("convex", COUNT)
        is_circle, circle_points = fit_arc(path)
        print(is_circle)
        if is_circle:
            return circle_points
    return path

def is_convex(points, tolerance=15):
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return np.all(np.abs(np.mean(hull_points, axis=0) - np.mean(points, axis=0)) <= tolerance)
    except QhullError:
        return False

# Function to fit a line to points and check if they form a straight line
def fit_line(points, tolerance=20, x_threshold=15):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    x_values = points[:, 0]
    y_values = points[:, 1]
# Check if all x values are the same within the threshold
    if np.max(x_values) - np.min(x_values) < x_threshold:
        # All x values are within the threshold, indicating a vertical line
        return True, np.column_stack((np.mean(x_values) * np.ones_like(y_values), y_values))

    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    residuals = np.abs(y - y_pred)
    return np.all(residuals <= tolerance), np.column_stack((X.flatten(), y_pred))

def fit_arc(points, tolerance=2):
    def calc_R(xc, yc): #radius
        return np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
    
    def cost(params): # residuals for the least squares optimization.
        xc, yc, r = params
        return calc_R(xc, yc) - r
    
    # Initial guess for circle parameters
    x_m, y_m = np.mean(points, axis=0)
    r0 = np.mean(calc_R(x_m, y_m))
    initial_guess = [x_m, y_m, r0]
    
    # Perform least squares fitting
    result = least_squares(cost, initial_guess)
    xc, yc, r = result.x
    
    # Calculate residuals and check if they are within tolerance
    residuals = np.abs(calc_R(xc, yc) - r)
    # print(residuals)
    if np.all(residuals <= tolerance):
        angles = np.arctan2(points[:, 1] - yc, points[:, 0] - xc)
        angles = np.mod(angles + 2 * np.pi, 2 * np.pi)
        min_angle = np.min(angles)
        max_angle = np.max(angles)
        
        # Compute the angular span of the arc
        arc_span = max_angle - min_angle if max_angle - min_angle < np.pi else 2 * np.pi - (min_angle - max_angle)
        # print(arc_span)
        # Check if the arc span indicates a full circle or a partial arc
        is_circle = False
        if arc_span >= 2 * np.pi - 0.1:
            is_circle=True
        if is_circle:
            # If it's a circle, return all points on the circle
            circle_points = np.column_stack((xc + r * np.cos(angles), yc + r * np.sin(angles)))
        else:
            # If it's an arc, return only the arc points
            arc_points = np.column_stack((xc + r * np.cos(angles), yc + r * np.sin(angles)))
            return True, arc_points
        
        return is_circle, circle_points
    
    return False, None

def is_circle(points, tolerance=30):
    # Check if all points are equidistant from the center
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    return np.all(np.abs(radii - np.mean(radii)) <= tolerance)

def is_rectangle(points, tolerance=0.05):
    if len(points) != 4:
        return False
    vectors = np.diff(points, axis=0, append=points[:1])
    angles = np.arccos(np.clip(np.dot(vectors, np.roll(vectors, shift=-1, axis=0).T) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(np.roll(vectors, shift=-1, axis=0), axis=1)), -1.0, 1.0))
    right_angles = np.isclose(angles, np.pi / 2, atol=tolerance)
    return np.all(right_angles)

def is_regular_polygon(points, tolerance=0.05):
    if len(points) < 3:
        return False
    vectors = np.diff(points, axis=0, append=points[:1])
    side_lengths = np.linalg.norm(vectors, axis=1)
    return np.all(np.abs(side_lengths - np.mean(side_lengths)) <= tolerance)


def recognize_shape(regularized_shape):
    combined_points = np.vstack(regularized_shape)
    
    if is_convex(combined_points):
        is_circle, circle_points = fit_arc(combined_points)
        if is_circle:
            return "Circle"
        else:
            is_arc, arc_points = fit_arc(combined_points) # Adjust tolerance if needed
            if is_arc:
                return "Arc"
    else:
        is_line, _ = fit_line(combined_points)
        if is_line:

            return "Straight Line"
    
    return "Unknown Shape"


# Function to plot regularized shapes
def plot_regularized_shapes(regularized_shapes, connections=None):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k','orange', 'purple', 'brown', 'pink', 'grey']
    for i, shape in enumerate(regularized_shapes):
        c = colours[i % len(colours)]
        for path in shape:
            ax.plot(path[:, 0], path[:, 1], c=c, linewidth=2)
        if connections:
            for (line_i, conn_i) in connections[i]:
                if conn_i != -1:
                    line = shape[line_i]
                    conn_line = shape[conn_i]
                    ax.plot([line[-1, 0], conn_line[0, 0]], [line[-1, 1], conn_line[0, 1]], 'k--')
        ax.plot([], [], c=c, label=f"Shape {i+1}")
    ax.set_aspect('equal')
    ax.legend(loc='best')
    plt.show()

# Function to check if two points are nearly equal
def are_points_connected(p1, p2, tolerance=ENDPOINT_TOLERANCE):
    return np.linalg.norm(p1 - p2) < tolerance

# Function to check if end points of curves are connected
def check_connections(shapes):
    connections = []
    for shape in shapes:
        shape_connections = []
        for i, path in enumerate(shape):
            start_point = path[0]
            end_point = path[-1]
            connected = False
            for j, other_path in enumerate(shape):
                if i != j:
                    other_start_point = other_path[0]
                    other_end_point = other_path[-1]
                    if are_points_connected(end_point, other_start_point) or \
                       are_points_connected(start_point, other_end_point):
                        connected = True
                        shape_connections.append((i, j))
            if not connected:
                shape_connections.append((i, -1))  # Path is not connected to any other path
        connections.append(shape_connections)
    return connections

# Function to combine straight lines to form polygons
def combine_lines_to_polygon(lines, tolerance=ENDPOINT_TOLERANCE):
    combined_lines = []
    visited = set()
    for i, line in enumerate(lines):
        if i in visited:
            continue
        path = [line]
        visited.add(i)
        start_point = line[0]
        end_point = line[-1]
        for j, other_line in enumerate(lines):
            if j not in visited:
                if are_points_connected(end_point, other_line[0]):
                    path.append(other_line)
                    visited.add(j)
                    end_point = other_line[-1]
                elif are_points_connected(start_point, other_line[-1]):
                    path.insert(0, other_line)
                    visited.add(j)
                    start_point = other_line[0]
        combined_lines.append(np.vstack(path))
    return combined_lines


# Function to combine arcs to form a circle
def combine_arcs_to_circle(arcs):
    if len(arcs) == 0:
        print("No arcs to combine.")
        return False
    
    combined_points = np.vstack(arcs)
    is_circle, _ = fit_arc(combined_points)
    return is_circle

# Example usage
def main(csv_path):
    # Read CSV file and parse paths
    shapes = read_csv(csv_path)
     # Optionally visualize raw data
    # paths_XYs = read_csv_o(csv_path)
    # img_plot = plot(paths_XYs)

    # Regularize each shape
    
    regularized_shapes = [regularize_shape(shape) for shape in shapes]

    # Check connections between paths in each shape
    connections = check_connections(regularized_shapes)

     # Combine lines and arcs from regularized shapes
    combined_lines = [path for shape in regularized_shapes for path in shape if fit_line(path)[0]]
    polygon = combine_lines_to_polygon(combined_lines)

    # Combine arcs and check if they form a circle
    combined_arcs = [path for shape in regularized_shapes for path in shape if not fit_line(path)[0] and fit_arc(path)[0]]
    if combine_arcs_to_circle(combined_arcs):
        print("The combined arcs form a circle.")

    # Identify and recognize each regularized shape
    for i, regularized_shape in enumerate(regularized_shapes):
        shape_type = recognize_shape(regularized_shape)
        print(f"Regularized Shape {i+1} is identified as: {shape_type}")

    # Plot regularized shapes
    plot_regularized_shapes(regularized_shapes, connections)


# Replace with your CSV file path
csv_path = "C:/Users/Dell/Downloads/problems/problems/frag0.csv"
main(csv_path)







