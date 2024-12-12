import argparse
import sys
import time as t

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (x, y) for cardinal direction.
        :param name: common name of said direction.
        """
        self.direction = direction
        self.name = name


# 8 defined directions for sgbm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')


class Paths:
    def __init__(self):
        """
        represent the relation between the directions.
        """
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(self, max_disparity=64, P1=5, P2=70, csize=(7, 7), bsize=(3, 3)):
        """
        represent all parameters used in the sgm algorithm.
        :param max_disparity: maximum distance between the same pixel in both images.
        :param P1: penalty for disparity difference = 1
        :param P2: penalty for disparity difference > 1
        :param csize: size of the kernel for the census transform.
        :param bsize: size of the kernel for blurring the images and median filtering.
        """
        self.max_disparity = max_disparity
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize


def load_images(left_name, right_name, parameters):
    """
    read and blur stereo image pair.
    :param left_name: name of the left image.
    :param right_name: name of the right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: blurred left and right images.
    """
    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, parameters.bsize, 0, 0)
    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, parameters.bsize, 0, 0)
    return left, right


def get_indices(offset, dim, direction, height):
    """
    for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
    :param offset: difference with the main diagonal of the cost volume.
    :param dim: number of elements along the path.
    :param direction: current aggregation direction.
    :param height: H of the cost volume.
    :return: arrays for the y (H dimension) and x (W dimension) indices.
    """
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset, parameters):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
    penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume, parameters, paths):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)

    path_id = 0
    for path in paths.effective_paths:
        print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')
        sys.stdout.flush()
        dawn = t.time()

        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, parameters), axis=0)

        if main.direction == E.direction:
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, parameters), axis=0)

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, parameters)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, parameters)

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, parameters)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, parameters)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return aggregation_volume


def compute_costs(left, right, parameters, save_images,grid):
    """
    first step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    height = left.shape[0]
    width = left.shape[1]
    cheight = parameters.csize[0]
    cwidth = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity

    left_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    right_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)

    print('\tComputing left and right census...', end='')
    sys.stdout.flush()
    dawn = t.time()
    # pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            left_census = np.int64(0)
            center_pixel = left[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        left_census = left_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        left_census = left_census | bit
            left_img_census[y, x] = np.uint8(left_census)
            left_census_values[y, x] = left_census

            right_census = np.int64(0)
            center_pixel = right[y, x]
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        right_census = right_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        right_census = right_census | bit
            right_img_census[y, x] = np.uint8(right_census)
            right_census_values[y, x] = right_census

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    if save_images:
        cv2.imwrite('left_census.png', left_img_census)
        cv2.imwrite('right_census.png', right_img_census)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, disparity):
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_xor = np.int64(np.bitwise_xor(np.int64(right_census_values), lcensus))
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        right_cost_volume[:, :, d] = right_distance

        
    # grid = 4
    if(grid != 1):
        height,width,disparity = right_cost_volume.shape

        h = (height+grid - 1)// grid
        w = (width+grid - 1)// grid
        result_left = np.zeros((h, w,disparity), dtype=left_cost_volume.dtype)
        result_right = np.zeros((h, w,disparity), dtype=right_cost_volume.dtype)

        for d in range(0,disparity):       

            for i in range(h):
                for j in range(w):

                    start_row, end_row = i * grid, min((i + 1) * grid, height)
                    start_col, end_col = j * grid, min((j + 1) * grid, width)

                    result_left[i, j,d] = np.sum(left_cost_volume[start_row:end_row, start_col:end_col,d])
                    result_right[i, j,d] = np.sum(right_cost_volume[start_row:end_row, start_col:end_col,d])



        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))
        print(left_cost_volume.shape)
        print(result_left.shape)

        return result_left,result_right     
       
    return left_cost_volume, right_cost_volume


def select_disparity(aggregation_volume,height,width,grid):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """

    print(aggregation_volume.shape)
    volume = np.sum(aggregation_volume, axis=3)
    h = (height+grid - 1)// grid
    w = (width+grid - 1)// grid
    h = h*grid
    w = w*grid
    final = np.zeros((h,w))
    disparity_map = np.argmin(volume, axis=2)
    for i in range(h):
        for j in range(w):
            final[i][j] = disparity_map[i//grid][j//grid]
    print(final.shape)          
    actual = final[0:height,0:width]
    print(actual.shape)  
    return actual


def normalize(volume, parameters):
    """
    transforms values from the range (0, 64) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 255.0 * volume / parameters.max_disparity


def get_accuracy(disparity, gt, disp = 64):
    """
    computes the accuracy of the disparity map.
    :param disparity: disparity image.
    :param gt: path to ground-truth image.
    :param args: program arguments.
    :return: rate of correct predictions.
    """
    gt = np.float32(cv2.imread(gt, cv2.IMREAD_GRAYSCALE))
    gt = np.int16(gt / 255.0 * float(disp))
    disparity = np.int16(np.float32(disparity) / 255.0 * float(disp))
    correct = np.count_nonzero(np.abs(disparity - gt) <= 3)
    return float(correct) / gt.size



def compute_sgbm_time_accuracy(left_name="../data/cones/im2.png", right_name="../data/cones/im6.png",
                          left_gt_name='../data/cones/disp2.png', right_gt_name='../data/cones/disp6.png', disparity=64,
                          save_images=False, evaluation=True):
    

    parameters = Parameters(max_disparity=disparity, P1=10, P2=120, csize=(7, 7), bsize=(3, 3))

    paths = Paths()
    print('\nLoading images...')
    left, right = load_images(left_name, right_name, parameters)

    grid_sizes = list(range(1, 11))  
    time = []
    accuracy_left = []
    accuracy_right = []

    for grid in grid_sizes:
        
        dawn = t.time()
        print(f'\nStarting cost computation for grid size {grid}...')
        left_cost_volume, right_cost_volume = compute_costs(left, right, parameters, save_images, grid)
        
        if save_images:
            left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), parameters))
            cv2.imwrite(f'disp_map_left_cost_volume_{grid}.png', left_disparity_map)
            right_disparity_map = np.uint8(normalize(np.argmin(right_cost_volume, axis=2), parameters))
            cv2.imwrite(f'disp_map_right_cost_volume_{grid}.png', right_disparity_map)

        print(f'\nStarting left aggregation computation for grid size {grid}...')
        left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)

        print(f'\nStarting right aggregation computation for grid size {grid}...')
        right_aggregation_volume = aggregate_costs(right_cost_volume, parameters, paths)

        print(f'\nSelecting best disparities for grid size {grid}...')
        left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume, left.shape[0], left.shape[1], grid), parameters))
        right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume, left.shape[0], left.shape[1], grid), parameters))

        print(f'\nApplying median filter for grid size {grid}...')
        left_disparity_map = cv2.medianBlur(left_disparity_map, parameters.bsize[0])
        right_disparity_map = cv2.medianBlur(right_disparity_map, parameters.bsize[0])

        if evaluation:
            print(f'\nEvaluating left disparity map for grid size {grid}...')
            accuracy = get_accuracy(left_disparity_map, left_gt_name)
            accuracy_left.append(accuracy)
            print(f'\taccuracy = {accuracy * 100.0:.2f}%')

            print(f'\nEvaluating right disparity map for grid size {grid}...')
            accuracy = get_accuracy(right_disparity_map, right_gt_name)
            accuracy_right.append(accuracy)
            print(f'\taccuracy = {accuracy * 100.0:.2f}%')

        dusk = t.time()
        print(f'\nTotal execution time for grid size {grid} = {dusk - dawn:.2f}s')
        time.append(dusk - dawn)

    return grid_sizes, accuracy_left, accuracy_right, time



def plot_results(grid_sizes, accuracy_left, accuracy_right, time, plot_save=True):
    """
    Just a plotter code to help visualise 
    """

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot accuracy on the left y-axis
    ax1.set_xlabel('Grid Size')
    ax1.set_ylabel('Accuracy', color='blue')
    ax1.plot(grid_sizes, accuracy_left, marker='o', color='blue', label='Left Disparity Accuracy')
    ax1.plot(grid_sizes, accuracy_right, marker='s', color='cyan', label='Right Disparity Accuracy')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc="upper left")

    # Create a second y-axis for time taken
    ax2 = ax1.twinx()
    ax2.set_ylabel('Time Taken (s)', color='purple')
    ax2.plot(grid_sizes, time, marker='d', color='purple', label='Time Taken')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(loc="upper right")

    plt.title('Accuracy and Time Taken vs Grid Size')
    fig.tight_layout()

    if plot_save:
        plt.savefig('../results/accuracy_and_time_vs_grid_size.png')

    plt.show()

# do it on cone 
grid_sizes, accuracy_left, accuracy_right, time = compute_sgbm_time_accuracy()


plot_results(grid_sizes, accuracy_left, accuracy_right, time)
