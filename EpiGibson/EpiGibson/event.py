from .action import *
from .utils import *
import math
import cv2

class idleEvent(Event):

    def __init__(self, agent:Agent, idleTime=10):
        super().__init__(event='idle', action_list=[idleAction(agent, idleTime)])

class brushTeethEvent(Event):

    def __init__(self, env:og.Environment, agent:Agent, teethbrush:Object):
        grasp_action = graspAction(teethbrush, agent=agent, agent_angle_offset=-np.pi / 4, description="grasp the teethbrush")
        inter_action = interAction(teethbrush, agent=agent, description="brush teeth")
        super().__init__(event='brushTeeth',
            action_list=[
                walkAction(env, agent, teethbrush.get_pos(), description="walk to location of teethbrush"),
                watchAction(agent, teethbrush.get_pos(), description="watch to teethbrush"),
                [grasp_action, idleAction(agent)],
                [inter_action, idleAction(agent)],
                [reverseAction(inter_action, description="brush teeth"), idleAction(agent)],
                [reverseAction(grasp_action, description="put the teethbrush back in place"), idleAction(agent)],
            ],
            description='brush teeth'
        )
        self.agent = agent
    
    def pre_event(self):
        self.agent.set_grasp_offset(front=0.05, left=-0.05, height=0.02)

class cookEvent(Event):

    def __init__(self, env:og.Environment, agent:Agent, objects:Dict):
        stove_pos = objects['stove'].get_pos()
        stove_pos[0] -= 0.1
        stove_pos[1] -= 0.34
        stove_pos[2] = 1.1
        bowl_pos = objects['bowl'].get_pos()
        bowl_pos[2] += 0.15
        action_list = [
            walkAction(env, agent, objects['stove'].get_pos(), description='walk to stove'),
            watchAction(agent, objects['stove'].get_pos(), description='watch to stove'),
            [toggleAction(objects['stove'], True, description='turn on the stove'), idleAction(agent, idleTime=40)],
        ]
        for idx, object in enumerate([objects['baking_sheet']] + list(objects['food'].values())):
            desc = 'put baking sheet onto the stove' if idx == 0 else 'put foods from bowl to baking sheet'
            action_list += [
                watchAction(agent, object.get_pos(), description=desc),
                [moveAction(object, object.get_pos() + torch.tensor([0,0,0.1]), enable_gravity=False, description=desc),idleAction(agent)],
                watchAction(agent, stove_pos, description=desc),
                [moveAction(object, stove_pos, enable_gravity=False, description=desc), idleAction(agent)],
                [moveAction(object, stove_pos - torch.tensor([0,0,0.1]) + (idx==2) * torch.tensor([0.03,0.03,0]), enable_gravity=True, description=desc),idleAction(agent)],
            ]
        action_list.append(idleAction(agent, idleTime=40, description='cook the food'))
        
        for object in list(objects['food'].values()):
            desc = 'put food from baking sheet to bowl'
            action_list += [
                watchAction(agent, stove_pos, description=desc),
                [moveAction(object, stove_pos, enable_gravity=False, description=desc), idleAction(agent)],
                watchAction(agent, bowl_pos, description=desc), 
                [moveAction(object, bowl_pos, enable_gravity=False, description=desc), idleAction(agent)],
                [moveAction(object, bowl_pos - torch.tensor([0,0,0.1]), enable_gravity=True, description=desc), idleAction(agent)],
            ]
        
        action_list += [
            watchAction(agent, stove_pos, description='watch to the stove'),
            [toggleAction(objects['stove'], False, 'turn off the stove'), idleAction(agent, idleTime=40)],
        ]

        super().__init__(event='cook',
            action_list=action_list,
            description='cook food'
        )

class eatEvent(Event):

    def __init__(self, env:og.Environment, agent:Agent, objects:Dict):
        self.agent = agent
        action_list = []
        for object in list(objects['food'].values()):
            action_list += [
                walkAction(env, agent, object.get_pos(), description='walk to food'),
                watchAction(agent, object.get_pos(), description='watch to food'),
                [graspAction(object, agent=agent, description='grasp the food'), idleAction(agent)],
                [interAction(object, agent=agent, description='eat the food'), idleAction(agent)],
                [eatAction(object, agent=agent, description='eat the food'), idleAction(agent)],
            ]
        
        super().__init__(event='eat',
            action_list=action_list,
            description='eat food'
        )

    def pre_event(self):
        self.agent.set_grasp_offset(front=0.05, left=-0.05, height=0.05)


class washFaceEvent(Event):

    def __init__(self, env:og.Environment, agent:Agent, towel:Object):
        grasp_action = graspAction(towel, agent=agent, description='grasp the towel')
        inter_action = interAction(towel, agent=agent, description='wash face by towel')
        super().__init__(event='washFace',
            action_list=[
                walkAction(env, agent, towel.get_pos(), description='walk to location of the towel'),
                watchAction(agent, towel.get_pos(), description='watch to the towel'),
                [grasp_action, idleAction(agent)],
                [inter_action, idleAction(agent)],
                [reverseAction(inter_action, description='wash face by towel'), idleAction(agent)],
                [reverseAction(grasp_action, description='put the towel back in place'), idleAction(agent)],
            ],
            description='wash face'
        )
        self.agent = agent
    
    def pre_event(self):
        self.agent.set_grasp_offset(front=0.05, left=-0.05, height=0.02)

class cleanEvent(Event):

    def __init__(self, env: og.Environment, agent: Agent, broom:Object, seg_map, room_type):
        self.agent = agent
        self.hold_action = holdAction(entity=broom, agent=agent)

        robot_chassis_extent = agent.entity.reset_joint_pos_aabb_extent[:2]
        radius = torch.norm(robot_chassis_extent) / 2.0 + 0.3
        radius_pixel = int(math.ceil(radius / env.scene.trav_map.map_resolution))
        trav_map = torch.tensor(cv2.erode(1.0*(env.scene.trav_map.floor_map[0]>0).cpu().numpy(), torch.ones((radius_pixel, radius_pixel)).cpu().numpy()))

        point_list = [seg_map.get_random_point_by_room_type(room_type, mask=trav_map)[1] for i in range(5)]
        super().__init__(
            event='clean',
            action_list=[
                walkAction(env, agent, broom.get_pos(), description='walk to location of the broom'), 
                watchAction(agent, broom.get_pos(), description='watch to the broom'),
                [graspAction(broom, agent=agent, description='grasp the broom'), idleAction(agent)],
            ] + 
            [
                [walkAction(env, agent, point, description='clean the house'), self.hold_action]
                for point in point_list
            ] + 
            [
                [walkAction(env, agent, broom.get_pos(), description='walk to broom\'s original position'), self.hold_action],
                [watchAction(agent, broom.get_pos(), description='walk to broom\'s original position'), self.hold_action],
                [moveAction(broom, broom.get_pos(), description='put the broom back in place'), idleAction(agent)]
            ],
            description='clean house'
        )
    
    def pre_event(self):
        self.agent.set_grasp_offset(front=0.05, left=-0.1, height=-0.45)

class readBookEvent(Event):

    def __init__(self, env:og.Environment, agent:Agent, book:Object):
        grasp_action = graspAction(book, agent=agent, description='pick up the book')
        inter_action = interAction(book, agent=agent, velocity=0.002, max_delta=0.02, description='read the book')
        super().__init__(event='readBook',
            action_list=[
                walkAction(env, agent, book.get_pos(), description='walk to location of the book'),
                watchAction(agent, book.get_pos(), description='watch to the book'),
                [grasp_action, idleAction(agent)],
                [inter_action, idleAction(agent)],
                [reverseAction(inter_action, description=inter_action.description), idleAction(agent)],
                [reverseAction(grasp_action, description='put the book back in place'), idleAction(agent)],
            ],
            description='read book'
        )
        self.agent = agent
    
    def pre_event(self):
        self.agent.set_grasp_offset(front=0.05, left=-0.05, height=0.03)
