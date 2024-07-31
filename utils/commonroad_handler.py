from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import State, InitialState, PMState, KSState
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad_crime.data_structure.configuration import CriMeConfiguration
from commonroad_crime.data_structure.crime_interface import CriMeInterface
from commonroad_crime.measure import TTCStar, TTR
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50

class CommonRoadHandler:
    EGO_ID = 100
    EGO_X = 80 # fix the x position for better display
    TOTAL_TIME = 3 # seconds
    TIME_STEP_SIZE = 0.25
    TOTAL_TIME_STEP = int(TOTAL_TIME/TIME_STEP_SIZE)

    def __init__(self, scenario_path ,debug=False):
        self.debug = debug
        self.scenario_path = scenario_path
        self.reset()

    def reset(self):
        self.scenario, self.planning_problem_set = CommonRoadFileReader(self.scenario_path).open(lanelet_assignment=True)
        self.ego = self.scenario.obstacle_by_id(self.EGO_ID)

    def get_min_ttr(self, obs, ego_position):
        ego_obs = obs[0]
        ego_y = ego_position[1]
        self.ego.initial_state = InitialState(
            position=np.array([self.EGO_X, ego_y]),
            orientation=np.arctan(ego_obs[4]/ego_obs[3]),
            velocity=np.sqrt(ego_obs[3]**2 + ego_obs[4]**2),
            time_step=0,
            acceleration=0,
        )

        npcs = []

        for i, this_obs in enumerate(obs[1:]):
            if this_obs[0] == 0:
                continue
            new_id = self.scenario.generate_object_id()
            new_npc = DynamicObstacle(
                obstacle_id=new_id,
                obstacle_type=self.ego.obstacle_type,
                obstacle_shape=self.ego.obstacle_shape,
                initial_state=InitialState(
                    position=np.array([self.EGO_X + this_obs[1], ego_y + this_obs[2]]),
                    orientation=np.arctan(this_obs[4]/this_obs[3]),
                    velocity=np.sqrt(this_obs[3]**2 + this_obs[4]**2),
                    time_step=0,
                    acceleration=0,
                )
            )
            npcs.append(new_npc)
            self.scenario.add_objects(new_npc)
            if self.debug:
                print(new_npc.initial_state)

        self.ego.prediction = self._generate_trajectory_prediction(self.ego)
        for npc in npcs:
            npc.prediction = self._generate_trajectory_prediction(npc)
        self.scenario.assign_obstacles_to_lanelets() # Necceary for further computations

        config = CriMeConfiguration()
        config.update(
            ego_id = self.EGO_ID,
            sce = self.scenario
        )
        if self.debug:
            config.print_configuration_summary()

        evaluator = TTR(config)
        final_ttr = evaluator.compute(0)

        return final_ttr

        


    def _generate_trajectory_prediction(self, dynamic_obstacle: DynamicObstacle) -> TrajectoryPrediction:
        states = []
        initial_state = dynamic_obstacle.initial_state
        velocity = initial_state.velocity
        orientation=initial_state.orientation # rad
        vx = velocity*np.cos(orientation)
        vy = velocity*np.sin(orientation)
        if self.debug:
            print(f'v:{velocity}, vx:{vx}, vy:{vy}, orientation:{orientation}')
        for i in range(self.TOTAL_TIME_STEP):
            new_state = KSState(
                time_step=initial_state.time_step + i,
                position=np.array([initial_state.position[0] + vx*i*self.TIME_STEP_SIZE, initial_state.position[1] + vy*i*self.TIME_STEP_SIZE]),
                velocity=velocity,
                orientation=orientation
            )
            states.append(new_state)
        trajectory = Trajectory(initial_state.time_step, states)
        prediction = TrajectoryPrediction(trajectory, dynamic_obstacle.obstacle_shape)
        return prediction
    
    def _visualize(self, evaluator):
        evaluator.visualize()
    

if __name__ == '__main__':
    scenario_path = '/home/tay/Workspace/Coverage/PhysicalCoverage/environments/highway/highway.xml'
    handler = CommonRoadHandler(scenario_path, debug=True)
    obs = [
        [1, 123, 8, 30, 0, -1, 2],
        [1, -4.4398, -4, 24, 0, 6, 1],
        [1, 9.2, 4, 24.7, 0, 8, 3],
        [1, 26.6008,  -8.,      23.4417,   3.,       5.,       0.],
        [1, -67.2306,   0.0003,  18.9922,  -0.157,    5.,       2.],
        [  0.,0.,0.,0.,0.,0.,0.    ],
        [  0.,0.,0.,0.,0.,0.,0.    ],
        [  0.,0.,0.,0.,0.,0.,0.    ],
        [  0.,0.,0.,0.,0.,0.,0.    ]
    ]
    minttr = handler.get_min_ttr(obs)
    print(f'Min TTR: {minttr}')