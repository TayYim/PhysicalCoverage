import numpy as np

def find_points_on_side(edge_length, x1, y1, x2, y2, side='left'):
    # 步骤1：定义正方形区域内的整数坐标点
    points = [(x, y) for x in range(-edge_length, edge_length+1) for y in range(-edge_length, edge_length+1)]
    
    # 步骤2：计算直线的斜率和截距
    if x2 - x1 == 0:  # 垂直线的特殊情况
        m = float('inf')
        b = x1
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
    
    # 步骤3：判断每个点是否在直线的side
    result_points = []
    if side == 'left':
        for x, y in points:
            if m == float('inf'):  # 垂直线的特殊情况
                if x < b:
                    result_points.append((x, y))
            elif y > m * x + b:
                result_points.append((x, y))
    else:
        for x, y in points:
            if m == float('inf'):  # 垂直线的特殊情况
                if x > b:
                    result_points.append((x, y))
            elif y < m * x + b:  # 修改这里的判断条件
                result_points.append((x, y))

    
    # 步骤4：返回所有在直线左侧的点
    return result_points

def get_points(edge_length, tracked_objects, lane_positions):
    upperlane = lane_positions[0]
    lowerlane = lane_positions[1]

    # x,y 相反
    left_points = find_points_on_side(edge_length, 
                                    upperlane[0][1], 
                                    upperlane[0][0], 
                                    upperlane[1][1], 
                                    upperlane[1][0],
                                    'left')
    
    right_points = find_points_on_side(edge_length,
                                        lowerlane[0][1],
                                        lowerlane[0][0],
                                        lowerlane[1][1],
                                        lowerlane[1][0],
                                        'right')
    
    l = [(item.position[0][1], item.position[0][0]) for item in tracked_objects] + left_points + right_points

    return l

def compute_grid(edge_length, cell_size, points):

    grid_size = edge_length // cell_size
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # 将坐标点映射到网格上
    for (x, y) in points:
        if -edge_length//2 <= x <= edge_length//2 and -edge_length//2 <= y <= edge_length//2:
            # 计算坐标点对应的网格单元
            grid_x = int((x + edge_length//2) // cell_size)
            grid_y = int((y + edge_length//2) // cell_size)
            grid_y = grid_size - grid_y - 1

            
            
            # 确保索引在有效范围内
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                # print(f"({x}, {y})->({grid_y}, {grid_x})")
                grid[grid_y, grid_x] = 1

    return grid


def get_planning_type(car_controller):
    planning = 0 # -1: left, 0: straight, 1: right
    if car_controller.current_lane == car_controller.desired_lane:
        planning = 0
    elif car_controller.current_lane < car_controller.desired_lane:
        planning = 1
    else:
        planning = -1

    return planning
