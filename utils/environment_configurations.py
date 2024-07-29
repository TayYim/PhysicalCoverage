class HighwayKinematics:
    def __init__(self):
        self.steering_angle  = 30 # Deg
        self.max_velocity    = 30 # m/s

class BeamNGKinematics:
    def __init__(self):
        self.steering_angle  = 33 # Deg
        self.max_velocity    = 35 # m/s

class WaymoKinematics:
    def __init__(self):
        self.steering_angle  = 33 # Deg
        self.max_velocity    = 35 # m/s

class RRSConfig:
    def __init__(self, beam_count = 3):
        self.beam_count     = beam_count

class FingerprintConfig:
    def __init__(self, edge_length = 60, cell_size = 4):
        self.edge_length     = edge_length # m
        self.cell_size       = cell_size   # m

class PolarConfig:
    def __init__(self, lx = 12, ly = 30, n_rad = 4, n_ring = 3):
        self.lx     = lx
        self.ly     = ly
        self.n_rad  = n_rad
        self.n_ring = n_ring