import numpy as np
from src.task_generator import generate_linear_task, generate_circle_task, generate_moon_task
#from task_generator import generate_linear_task, generate_circle_task, generate_moon_task


def generate_grid_data(support_data:np.ndarray, num_coord:int=50) -> np.ndarray:
    
    min_x, min_y = list(np.min(support_data, axis=0))
    max_x, max_y = list(np.max(support_data, axis=0))
    
    x_coordinates = np.linspace(min_x-0.2, max_x+0.2, num_coord)
    y_coordinates = np.linspace(min_y-0.2, max_y+0.2, num_coord)
    
    mesh_x_coord, mesh_y_coord = np.meshgrid(x_coordinates, y_coordinates.transpose())
    
    res = []
    for i in range(num_coord):
        for j in range(num_coord):
            res.append([mesh_x_coord[i][j], mesh_y_coord[i][j]])
    return np.array(res)


def generate_N_dim_tasks(support_data:np.array, num_coord:int=50, num_dim:int=3, num_query:int=2500, random_seed:int=42) -> np.ndarray:

    """
    Given a set of N-dim data, generate a set of query data.
    
    Args:
        support_data (np.ndarray, shape=(N, D)): The in-context data.
        num_coord (int): The number of grids for each axis, default is 50.
        num_dim (int): The number of dimensions, default is 3.
        num_query (int): Number of random samples to generate, default is 2500. This parameter is only used when num_dim > 2.
        random_seed (int): Random seed for reproducibility, default is 42.
    Returns:
        query_data: np.ndarray
    """
    min_val_array = np.min(support_data, axis=0)
    max_val_array = np.max(support_data, axis=0)

    feature_single_dim_grid = [np.linspace(min_val, max_val, num_coord) for min_val, max_val in zip(min_val_array, max_val_array)]

    query_axes = np.meshgrid(*feature_single_dim_grid, indexing='ij')

    query_data = np.stack([grid.ravel() for grid in query_axes], axis=1)
    
    # Random sampling
    if num_dim > 2:
        rng = np.random.RandomState(random_seed)
        assert num_query <= query_data.shape[0], f"num_query must be less than or equal to the number of query data points"
        rng.shuffle(query_data)
        query_data = query_data[:num_query]
    else:
        query_data = query_data

    return query_data


if __name__ == "__main__":

    support_data, support_labels = generate_linear_task(num_classes=2, num_samples=128, num_feat=3, mode="train", randseed=11)
    query_data = generate_N_dim_tasks(support_data, num_coord=50, num_dim=3)
    print(np.array(query_data).shape)