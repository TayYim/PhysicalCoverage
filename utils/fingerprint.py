import numpy as np

def find_points_on_side(lx, ly, x1, y1, x2, y2, side='left'):
    # 步骤1：定义区域内的整数坐标点
    points = [(x, y) for x in range(-lx//2, lx//2+1) for y in range(-ly//2, ly//2+1)]
    
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

def get_points(lx ,ly, tracked_objects, lane_positions, reverse=True):
    upperlane = lane_positions[0]
    lowerlane = lane_positions[1]

    if reverse:
        # x,y 相反
        left_points = find_points_on_side(lx, ly,
                                        upperlane[0][1], 
                                        upperlane[0][0], 
                                        upperlane[1][1], 
                                        upperlane[1][0],
                                        'left')
        
        right_points = find_points_on_side(lx, ly,
                                            lowerlane[0][1],
                                            lowerlane[0][0],
                                            lowerlane[1][1],
                                            lowerlane[1][0],
                                            'right')
        
        l = [(item.position[0][1], item.position[0][0]) for item in tracked_objects] + left_points + right_points
    else:
        left_points = find_points_on_side(lx, ly,
                                        upperlane[0][0], 
                                        upperlane[0][1], 
                                        upperlane[1][0], 
                                        upperlane[1][1],
                                        'left')
        
        right_points = find_points_on_side(lx, ly,
                                            lowerlane[0][0],
                                            lowerlane[0][1],
                                            lowerlane[1][0],
                                            lowerlane[1][1],
                                            'right')
        
        l = [(item.position[0][0], item.position[0][1]) for item in tracked_objects] + left_points + right_points

    return l

def get_points_from_raw_file(lx ,ly, tracked_objects, lane_positions, reverse=True):
    upperlane = lane_positions[0]
    lowerlane = lane_positions[1]

    if reverse:
        # x,y 相反
        left_points = find_points_on_side(lx, ly,
                                        upperlane[0][1], 
                                        upperlane[0][0], 
                                        upperlane[1][1], 
                                        upperlane[1][0],
                                        'left')
        
        right_points = find_points_on_side(lx, ly,
                                            lowerlane[0][1],
                                            lowerlane[0][0],
                                            lowerlane[1][1],
                                            lowerlane[1][0],
                                            'right')
        
        l = [(item[1], item[0]) for item in tracked_objects] + left_points + right_points
    else:
        left_points = find_points_on_side(lx, ly,
                                        upperlane[0][0], 
                                        upperlane[0][1], 
                                        upperlane[1][0], 
                                        upperlane[1][1],
                                        'left')
        
        right_points = find_points_on_side(lx, ly,
                                            lowerlane[0][0],
                                            lowerlane[0][1],
                                            lowerlane[1][0],
                                            lowerlane[1][1],
                                            'right')
        
        l = [(item[0], item[1]) for item in tracked_objects] + left_points + right_points

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

def compute_grid_polar(lx, ly, nrad, nring, points):

    matrix = np.zeros((nrad, nring), dtype=int)

    for (x, y) in points:

        if x==0 and y==0:
            continue

        if abs(x) > lx / 2 or abs(y) > ly / 2:
            continue

        # 计算点到原点的角度
        angle = np.arctan2(y, x)
        if angle < 0:
            angle += 2 * np.pi  # 将角度转换到 [0, 2π) 范围内

        # 计算半径线编号
        rad_index = int(np.floor(angle / (2 * np.pi / nrad))) % nrad

        # 计算该角度下半径线与边界的交点
        if abs(np.cos(angle)) * ly > abs(np.sin(angle)) * lx:
            # 与垂直边界相交
            max_r = abs(lx / (2 * np.cos(angle)))
        else:
            # 与水平边界相交
            max_r = abs(ly / (2 * np.sin(angle)))

        # 计算点到原点的距离
        r = np.sqrt(x**2 + y**2)

        # 计算点在哪个环内
        ring_index = int(np.floor(r / (max_r / nring)))

        # 确保 ring_index 不超过 nring-1
        ring_index = min(ring_index, nring-1)

        matrix[rad_index, ring_index] = 1

    return matrix


def get_planning_type(car_controller):
    planning = 0 # -1: left, 0: straight, 1: right
    if car_controller.current_lane == car_controller.desired_lane:
        planning = 0
    elif car_controller.current_lane < car_controller.desired_lane:
        planning = 1
    else:
        planning = -1

    return planning
