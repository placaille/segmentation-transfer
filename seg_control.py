#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

import os
import requests
import sys
import argparse
import pyglet
from pyglet.window import key
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from src.models.models import SegNet
from image_processing import logit_to_img
import torch
from matplotlib.figure import Figure
from multiprocessing import Process, Pipe
import multiprocessing as mp
from demo_utils import plotting
import time
from threading import Thread, Lock


# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand,
        frame_skip = args.frame_skip,
        distortion = args.distortion,
        seed = 0,
        camera_width = 160,
        camera_height = 120,
    )
else:
    env = gym.make(args.env_name)


def get_input_transform():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float()),
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),
        transforms.Lambda(lambda x: x.div_(255)),
    ])
    return transform

transform = get_input_transform()

# Register a keyboard handler
key_handler = key.KeyStateHandler()
lock = Lock()
out_obs = np.zeros((120, 160, 3))

net = SegNet(3, 4)
if not os.path.isfile('models/segnet.pth'):
    print('Model not found. Downloading')
    url = 'https://uc8f29bba686f8d63243e37138ab.dl.dropboxusercontent.com/cd/0/get/AXYFBov0NB2vIm3bZBlT-yEIBj13znBFOvggr-Sjk4tGUcZ_mDZcaUog1xIN3p_Tx-hPbY89m1tyou9qqio1ypyb1270aWLR5Ec7Nr3sK3uTnaKQSqwMDVvbW7WYN2zgGMA7uAkklJNKuqW-bWGgbylRhT8P8ggo_BjZVM6gtPme9fs_GZwqLss5o9dMDlpVgfc/file?_download_id=753895269774370474260145502354998998149572511748667472358993912&_notify_domain=www.dropbox.com&dl=1'
    r = requests.get(url, allow_redirects=True)
    open('models/segnet.pth', 'wb').write(r.content)

net.load('models/segnet.pth')
i = 0
comm1, comm_child1 = Pipe()
comm2, comm_child2 = Pipe()

def segment():
    global out_obs
    while True:
        with lock:
            obs = out_obs
        with torch.no_grad():
            d = torch.from_numpy(obs.transpose(2,0,1)).unsqueeze(0).type(torch.FloatTensor).div_(255)
            d = torch.from_numpy(obs)
            d = transform(d).unsqueeze(0)
            start = time.time()
            c = net(d).detach().argmax(1)
            start = time.time()
            img = logit_to_img(c)[0]
            comm1.send(obs)
            comm2.send(img)


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    global seg_img, i, out_obs
    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        print('done!')
        env.reset()
        env.render()

    with lock:
        out_obs = obs

    env.render()


def run_main():
    global env
    pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

    # Enter main event loop
    env.reset()
    env.render()

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        """
        This handler processes keyboard commands that
        control the simulation
        """

        if symbol == key.BACKSPACE or symbol == key.SLASH:
            print('RESET')
            env.reset()
            env.render()
        elif symbol == key.PAGEUP:
            env.unwrapped.cam_angle[0] = 0
        elif symbol == key.ESCAPE:
            env.close()
            sys.exit(0)
    env.unwrapped.window.push_handlers(key_handler)

        # Take a screenshot
        # UNCOMMENT IF NEEDED - Skimage dependency
        # elif symbol == key.RETURN:
        #     print('saving screenshot')
        #     img = env.render('rgb_array')
        #     save_img('screenshot.png', img)

    pyglet.app.run()

    env.close()

if __name__ == '__main__':
    mp.set_start_method('spawn')
    t = Thread(target=segment)
    child1 = Process(target=plotting.communicate, args=(comm_child1,))
    child1.start()
    child2 = Process(target=plotting.communicate, args=(comm_child2,))
    child2.start()
    t.start()
    run_main()

