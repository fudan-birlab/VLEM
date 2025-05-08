import os
import cv2
import yaml
import numpy as np
from datetime import datetime

import omnigibson as og
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True

from collections import OrderedDict

import numpy as np
from EpiGibson.event import *
from EpiGibson.utils import EventManager, Agent

def main():

    # environment configuration
    config_filename = "./config/env.yaml"
    cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # robot setting
    agent = Agent(env.robots[0])

    # camera setting
    viewer_camera = og.sim.viewer_camera
    viewer_camera.focal_length = 14.

    og.log.info("Resetting environment")
    env.reset()

    # object setting
    object_list = ['toothbrush', 'towel', 'broom', 'stove_igwqpj_0', 'sink_czyfhq_0', 'baking_sheet', 'bowl', 'bok_choy_1', 'bok_choy_2', 'fridge_dszchb_0', 'coffee_maker_pyttso_0', 'textbook']
    objects = {}

    for obj in env.scene.objects:
        if obj.name in object_list:
            objects[obj.name] = Object(obj)
        if 'door_lvgliq' in obj.name:
            obj.disable_gravity()
            obj.set_position(np.array([100 + np.random.randn(),100,100]))
    
    seg_map = env.scene.seg_map
    room_list = ['bathroom', 'bedroom', 'corridor', 'dining_room', 'kitchen', 'living_room', 'storage_room']
    eventManager = EventManager()

    eventManager.append(idleEvent(agent, idleTime=10))
    eventManager.append(brushTeethEvent(env, agent, objects['toothbrush']))
    eventManager.append(washFaceEvent(env, agent, objects['towel']))
    eventManager.append(cleanEvent(env, agent, objects['broom'], seg_map=seg_map, room_type=room_list[1]))
    eventManager.append(cookEvent(env, agent, {'stove':objects['stove_igwqpj_0'], 
                                               'baking_sheet':objects['baking_sheet'], 
                                               'bowl':objects['bowl'], 
                                               'food':{'bok_choy_1':objects['bok_choy_1'], 'bok_choy_2':objects['bok_choy_2']}}))
    eventManager.append(eatEvent(env, agent, {'food':{'bok_choy_1':objects['bok_choy_1'], 'bok_choy_2':objects['bok_choy_2']}}))
    eventManager.append(readBookEvent(env, agent, objects['textbook']))
    eventManager.append(idleEvent(agent, idleTime=10))

    agent.set_z_axis_parallel()

    action = np.zeros(11, dtype=np.float32)
    interval = 100
    action[3:7] = np.array([.4, 12., 12., 20.]) / interval
    for i in range(interval):
        donothing = OrderedDict([('robot0', action)])
        env.step(donothing)
    action *= 0
    donothing = OrderedDict([('robot0', action)])
    
    idx = 0
    room_list = []
    map_pos_list = []
    world_pos_list = []

    dataset_id = "0001"
    output_dir = f'./outputs/{dataset_id}-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}/'
    image_dir = output_dir + 'video_frames/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    while True:
        room_list.append(seg_map.get_room_type_by_point(agent.get_pos()[:2]))
        map_pos_list.append(seg_map.world_to_map(agent.get_pos()[:2]).numpy())
        world_pos_list.append(agent.get_pos()[:2].numpy())
        print(f'------ robot : {seg_map.get_room_type_by_point(agent.get_pos()[:2])}  viewer_camera : {seg_map.get_room_type_by_point(agent.get_pos()[:2])} ------')

        img = agent.get_obs()[0]['robot0:eyes:Camera:0']['rgb']
        cv2.imwrite(image_dir + f'{idx}.png', img.numpy()[:,:,[2,1,0]])
        idx += 1

        if eventManager.step():
            action_desc_list, event_desc_list = eventManager.all_description()
            np.save(output_dir + 'world_pos_list.npy', world_pos_list)
            np.save(output_dir + 'map_pos_list.npy', map_pos_list)
            np.save(output_dir + 'room_list.npy', room_list)
            np.save(output_dir + 'action_desc_list.npy', action_desc_list)
            np.save(output_dir + 'event_desc_list.npy', event_desc_list)

            last_action = ''
            ChatGPT_prompt = 'Describe what I am going to do based on my daily schedule below, within 50 words.\n'
            for action_desc in action_desc_list:
                if action_desc == last_action or action_desc == '':
                    continue
                else:
                    ChatGPT_prompt += action_desc + '\n'
                    last_action = action_desc
            with open(output_dir + 'ChatGPT_prompt.txt', 'w') as f:
                f.write(ChatGPT_prompt)

            print('------------------- finished. ------------------- ')
            break

        env.step(donothing)

    video = cv2.VideoWriter(output_dir + 'visual_input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1024,1024))
    for i in range(idx):
        video.write(cv2.imread(image_dir + f'{i}.png'))
    video.release()

    print(f'EpiGibson dataset generated in {output_dir}!')

    # Always close the environment at the end
    og.clear()

if __name__ == "__main__":
    main()
