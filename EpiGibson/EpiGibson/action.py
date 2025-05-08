import torch
from .utils import *

class idleAction(Action):

    def __init__(self, entity:Agent, idleTime=1, description=""):
        super().__init__(action='idle', entity=entity, description=description)
        self.idleTime = idleTime
    
    def pre_action(self):
        pos, ori = self.entity.get_pos_ori()
        pos[2] = 0.
        self.pos_list.append(pos)
        self.ori_list.append(ori)
    
    def step(self):
        if self.step_id == 0:
            self.pre_action()
        self.entity.set_pos(self.pos_list[0])
        self.entity.set_ori(self.ori_list[0])
        if self.step_id < self.idleTime:
            self.step_id += 1
        return self.step_id == self.idleTime

class walkAction(Action):

    def __init__(self, env:og.Environment, entity:Agent, pos:Union[List, np.array, torch.tensor], velocity=0.01, description=""):
        super().__init__(action='walk', entity=entity, description=description)
        self.env = env
        self.pos = pos
        self.velocity = velocity

    def pre_action(self):
        shortest_path = get_shortest_path(self.env, self.entity.get_pos(), self.pos, robot=self.entity.entity)
        self.pos_list = smooth_path(shortest_path, self.velocity)
        if len(self.pos_list) > 1:
            watch_action = watchAction(self.entity, target_dir=self.pos_list[1]-self.pos_list[0], angular_velocity=0.02)
            watch_action.pre_action()
            self.ori_list = watch_action.ori_list + points2quaternions(self.pos_list, initial_quaternion=watch_action.ori_list[-1])
            self.pos_list = watch_action.pos_list + self.pos_list
        else:
            self.ori_list = [self.entity.get_ori()]# * len(self.pos_list)
        for i in range(len(self.pos_list)):
            if len(self.pos_list[i]) == 2:
                self.pos_list[i] = torch.tensor([self.pos_list[i][0], self.pos_list[i][1], 0.])
            else:
                self.pos_list[i][2] = 0.

class moveAction(Action): # object move to pos

    def __init__(self, entity:Object, tgt_pos:Union[List, np.array, torch.tensor], velocity=0.01, enable_gravity=True, description=""):
        super().__init__(action='move', entity=entity, description=description)
        self.tgt_pos = torch.tensor(tgt_pos)
        self.velocity = velocity
        self.enable_gravity = enable_gravity

    def pre_action(self):
        src_pos, ori = self.entity.get_pos_ori()
        npoint = round((self.tgt_pos - src_pos).norm(p=2).item() // self.velocity + 1)
        vec = (self.tgt_pos - src_pos) / npoint
        self.pos_list = [src_pos + vec * i for i in range(npoint + 1)]
        self.ori_list = [ori] * len(self.pos_list)
        self.entity.disable_gravity()

    def post_action(self):
        if self.enable_gravity:
            self.entity.enable_gravity()

class reverseAction(Action):

    def __init__(self, raw_action:Action, description=""):
        super().__init__(action=raw_action.action+'_reverse', entity=raw_action.entity, description=description)
        self.raw_action = raw_action
    
    def pre_action(self):
        self.pos_list = self.raw_action.pos_list[::-1]
        self.ori_list = self.raw_action.ori_list[::-1]
    
    def post_action(self):
        if self.raw_action.action == 'grasp':
            self.entity.enable_gravity()

class graspAction(Action):

    def __init__(self, entity:Object, agent:Agent, velocity=0.01, agent_angle_offset=0., description=""):
        super().__init__(action='grasp', entity=entity, description=description)
        self.agent = agent
        self.velocity = velocity
        self.agent_angle_offset = agent_angle_offset

    def pre_action(self):
        tgt_pos = self.agent.grasp_pos()
        src_pos, ori = self.entity.get_pos_ori()
        npoint = round((tgt_pos - src_pos).norm(p=2).item() / self.velocity + 1)
        vec = (tgt_pos - src_pos) / npoint
        self.pos_list = [src_pos + vec * i for i in range(1, npoint + 1)]
        angle = self.agent.get_angle_z() + self.agent_angle_offset
        self.ori_list = [rotate_quaternion(ori, 'z', (i+1) * angle / len(self.pos_list)) for i in range(len(self.pos_list))]
        self.entity.disable_gravity()

class holdAction(Action):

    def __init__(self, entity:Object, agent:Agent, description=""):
        super().__init__(action='hold', entity=entity, description=description)
        self.agent = agent

    def pre_action(self):
        self.ori = self.entity.get_ori()
        self.step_id += 1
        self.init_angle = self.agent.get_angle_z()

    def step(self):
        if self.step_id == 0:
            self.pre_action()
        self.entity.set_pos(self.agent.grasp_pos())
        self.entity.set_ori(rotate_quaternion(self.ori, 'z', self.agent.get_angle_z() - self.init_angle))
        return True

class watchAction(Action):

    def __init__(self, entity:Agent, target_pos:Union[List, np.array, torch.tensor]=None, target_dir:Union[List, np.array, torch.tensor]=None, angular_velocity=0.1, description=""):
        assert not (target_pos is None and target_dir is None)
        super().__init__(action='watch', entity=entity, description=description)
        self.target_pos = target_pos
        self.target_dir = target_dir
        self.angular_velocity = angular_velocity
    
    def pre_action(self):
        pos, ori = self.entity.get_pos_ori()
        if self.target_dir is None:
            self.target_dir = np.array(self.target_pos)[:2] - np.array(pos)[:2]
        self.ori_list = calculate_quaternions_to_target(self.target_dir, ori, self.angular_velocity)
        pos[2] = 0.
        self.pos_list = [pos] * len(self.ori_list)

class eatAction(Action):

    def __init__(self, entity:Object, agent:Agent, description=""):
        super().__init__(action='eat', entity=entity, description=description)
        self.agent = agent

    def pre_action(self):
        src_pos, ori = self.entity.get_pos_ori()
        self.entity.disable_gravity()
        self.pos_list = [src_pos] * 10
        self.ori_list = [ori] * 10
        for i in range(len(src_pos)):
            src_pos[i] += 1e4 + 100 * torch.randn([])
        self.pos_list += [src_pos] * 10
        self.ori_list += [ori] * 10

class toggleAction(Action):

    def __init__(self, entity:Object, value=True, description=""):
        super().__init__(action='toggle', entity=entity, description=description)
        self.value = value
    
    def pre_action(self):
        states_status=reversed_unary_states['toggleable']
        self.entity.entity.states[states_status].set_value(self.value)

    def step(self):
        if self.step_id == 0:
            self.pre_action()
            self.step_id += 1
        return True

class unfoldAction(Action):

    def __init__(self, entity:Object, velocity=0.02, description=""):
        super().__init__(action='unfold', entity=entity, description=description)
        self.velocity = velocity
    
    def pre_action(self):
        states_status=reversed_unary_states['unfoldable']
        self.entity.entity.states[states_status].set_value(True)

    def step(self):
        if self.step_id == 0:
            self.pre_action()
            self.step_id += 1
        return True

class openAction(Action):

    def __init__(self, entity:Object, velocity=0.02, description=""):
        super().__init__(action='open', entity=entity, description=description)
        self.velocity = velocity

    def pre_action(self):
        self.oper_list = []
        oper = 0.
        while oper <= 1:
            self.oper_list.append(oper)
            oper += self.velocity
        self.oper_list.append(1.)

    def step(self):
        if self.step_id == 0:
            self.pre_action()
        if self.step_id < len(self.oper_list):
            states_status=reversed_unary_states['openable']
            self.entity.entity.states[states_status].set_value(self.oper_list[self.step_id])
            self.step_id += 1
        return self.step_id == len(self.oper_list)

class closeAction(Action):

    def __init__(self, entity:Object, velocity=0.02, description=""):
        super().__init__(action='close', entity=entity, description=description)
        self.velocity = velocity
    
    def pre_action(self):
        self.oper_list = []
        oper = 0.
        while oper <= 1:
            self.oper_list.append(oper)
            oper += self.velocity
        self.oper_list.append(1.)

    def step(self):
        if self.step_id == 0:
            self.pre_action()
        if self.step_id < len(self.oper_list):
            states_status=reversed_unary_states['openable']
            self.entity.entity.states[states_status].set_value(1. - self.oper_list[self.step_id])
            self.step_id += 1
        return self.step_id == len(self.oper_list)

class interAction(Action):

    def __init__(self, entity:Object, times=5, velocity=0.02, max_delta=0.2, agent=None, description=""):
        super().__init__(action='interaction', entity=entity, description=description)
        self.velocity = velocity
        self.times = times
        self.max_delta = max_delta
        if agent is not None:
            self.rotate_angle = quaternion2euler(agent.get_ori())[2] + np.pi
    
    def pre_action(self):
        pos, ori = self.entity.get_pos_ori()
        delta = torch.tensor([0., 0., self.velocity])
        offset = torch.tensor([0., 0., 0.])
        self.pos_list = []
        for i in range(self.times * 2):
            offset += delta
            while 0 < offset[2] and offset[2] < self.max_delta:
                self.pos_list.append(pos + offset)
                offset += delta
            delta *= -1
        self.ori_list = [rotate_quaternion(ori, 'z', self.rotate_angle / 20. * i) for i in range(20)] + [rotate_quaternion(ori, 'z', self.rotate_angle)] * len(self.pos_list)
        self.pos_list = [self.pos_list[0]] * 20 + self.pos_list
