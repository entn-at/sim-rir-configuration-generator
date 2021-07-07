#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Ref: Generation of large-scale simulated utterances in virtual rooms to 
    train deep-neural networks for far-field speech recognition in Google Home
"""
import math

import numpy as np


class Parameter(object):
    """Room configuration"""

    def __init__(self, room, mic, sound, t60):
        # room sizes
        self.room = room
        # micriphone positions
        self.mic = mic
        # reverberation time
        self.t60 = t60
        # sound source positions
        self.sound = sound

    @property
    def max_distance(self):
        """maximum distance"""
        return np.sqrt(np.sum((self.room-np.zeros_like(self.room))**2))

    @property
    def distance(self):
        """distance of the sound source to microphone"""
        return np.sqrt(np.sum((self.mic-self.sound)**2))

    def __str__(self):
        """string of parameter"""
        return "\tRoom={}\n\tMicPos={}\n\tSoundPos={}\n\tT60={}\n\tDistance={}\n".format(
            self.room, self.mic, self.sound, self.t60, self.distance)


class ParameterGenerator(object):
    """Room configuration generator

        The prior knowledge we have for simulation are the hardware characteristics like 
        microphone spacing, and targeted use-case scenarios like expected 
        room size distributions, reverberation times, background noise levels, 
        and target to microphone array distances. 
    """
    step = 0.5

    # The distribution of the distance from the target source to microphone.
    '''
    {
        1:14,
        2:22,
        3:29,
        4:21,
        5:9,
        6:2,
        7:1,
    }
    '''
    target_distance_distribute = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6,
        7
    ]

    # The range of the distance from the noise source to microphone.
    noise_distance_range = [0.5, 7.5]

    # The reverberation time (T60) distribution
    '''
    {
        0:4,
        0.1:6,
        0.2:7,
        0.3:10,
        0.4:13,
        0.5:17,
        0.6:17,
        0.7:13,
        0.8:7,
        0.9:4,
    }
    '''
    t60_distribute = [
        0.0, 0.0, 0.0, 0.0,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
        0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
        0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
        0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.9, 0.9, 0.9, 0.9
    ]

    def __init__(self,
                 sample_num,
                 seed,
                 config,
                 max_try_times=200):
        np.random.seed(seed)

        # room configurations number
        self.sample_num = sample_num

        # room sizes
        room = config['room']
        # The size of the room was randomly set to have 
        # a width uniformly between 3 meters to 10 meters, 
        # and a length between 3 meters to 8 meters 
        # and a height between 2.5 meters to 6 meters. 
        self.room_x_range = room['x']
        self.room_y_range = room['y']
        self.room_z_range = room['z']

        self.max_try_times = max_try_times

        target = config['target']
        # The azimuth and elevation are randomly selected to be in the 
        # interval [−180.0o , 180.0o ] and [45.0o,135.0o], 
        self.target_azimuth_range = target['azimuth']
        self.target_elevation_range = target['elevation']

        noise = config['noise'] 
        # number of noise sources
        self.noise_sources_number = noise['number']
        # The azimuth and elevation are randomly selected to be in the 
        # interval [−180.0o , 180.0o ] and [-30.0o,180.0o],
        self.noise_azimuth_range = noise['azimuth']
        self.noise_elevation_range = noise['elevation']

    def __iter__(self):
        for i in range(self.sample_num):
            # Generate target source rir parameters.
            target = self.gen_target()
            if target is None:
                yield None
                continue

            # Generate noise source rir parameters.
            noise_list = self.gen_noise_sources_list(
                target,
                self.noise_sources_number)
            if noise_list is None:
                yield None
                continue

            yield {
                'target_source': target,
                'noise_sources': noise_list,
            }

    def gen_target(self):
        """Generate target source rir parameters.
        """

        # The size of the room was randomly set to 
        # have a width uniformly between 3 meters to 10 meters, 
        # and a length between 3 meters to 8 meters and 
        # a height between 2.5 meters to 6 meters. 
        room = np.array(
            [
                np.random.uniform(
                    self.room_x_range[0],
                    self.room_x_range[1],
                ),
                np.random.uniform(
                    self.room_y_range[0],
                    self.room_y_range[1],
                ),
                np.random.uniform(
                    self.room_z_range[0],
                    self.room_z_range[1],
                )
            ]
        )

        mic = np.array(
            [
                np.random.uniform(0.0, 1.0)*room[0],
                np.random.uniform(0.0, 1.0)*room[1],
                np.random.uniform(0.0, 1.0)*room[2]
            ]
        )

        # reverberation times
        t60 = np.random.choice(ParameterGenerator.t60_distribute)

        # the target source locations are randomly selected with respect to the microphone. 
        sound = self.generate_sound_pos(
            room,
            mic,
            distance_distribute=ParameterGenerator.target_distance_distribute,
            azimuth_range=self.target_azimuth_range,
            elevation_range=self.target_elevation_range)

        if sound is None:
            return None

        return Parameter(room, mic, sound, t60)

    def gen_noise_sources_list(self, spk_rir_param, noise_sources_number):
        """Generate noise source rir parameters.
        """
        max_distance = spk_rir_param.max_distance

        stop = max_distance if max_distance <= ParameterGenerator.noise_distance_range[1] \
                else ParameterGenerator.noise_distance_range[1]

        noise_distance_distribute = np.arange(
            start=ParameterGenerator.noise_distance_range[0],
            stop=stop,
            step=ParameterGenerator.step)

        noise_sources = []
        for _ in range(noise_sources_number):
            # Randomly pick others sound sources.
            # the noise source locations are randomly selected with respect to the microphone. 
            sound = self.generate_sound_pos(
                spk_rir_param.room, spk_rir_param.mic,
                distance_distribute=noise_distance_distribute,
                azimuth_range=self.noise_azimuth_range,
                elevation_range=self.noise_elevation_range
            )
            if sound is None:
                return None

            rir = Parameter(
                spk_rir_param.room,
                spk_rir_param.mic,
                sound,
                spk_rir_param.t60)

            noise_sources.append(rir)

        return noise_sources

    def generate_sound_pos(
        self,
        room,
        micpos,
        distance_distribute,
        azimuth_range=[-180, 180],
        elevation_range=[45, 135]
    ):
        """ Sample a point within the sphere

            Ref: https://github.com/tomkocse/sim-rir-preparation/blob/master/prep_sim_rirs.sh
 
            https://ww2.mathworks.cn/help/matlab/ref/math_sphcart.png
                z
                |
                | elevation
                |  *
                |~/|
                |/ |
                .--|--------------- y
               / \ |
              /~~~\|
             / azimuth
            /
           x
        """
        def sph2cart(azimuth, elevation, r):
            x = r * np.cos(elevation) * np.cos(azimuth)
            y = r * np.cos(elevation) * np.sin(azimuth)
            z = r * np.sin(elevation)
            return x, y, z

        def angle_to_radian(x):
            return (x/180.)*math.pi

        def radian_to_angle(x):
            return (x)/math.pi*180.

        azimuth_range = [
            angle_to_radian(azimuth_range[0]),
            angle_to_radian(azimuth_range[1])
        ]

        elevation_range = [
            angle_to_radian(elevation_range[0]),
            angle_to_radian(elevation_range[1])
        ]
        space = ParameterGenerator.step

        for _ in range(self.max_try_times):
            # elevation
            elevation = np.random.uniform(
                elevation_range[0], elevation_range[1])

            # azimuth
            azimuth = np.random.uniform(
                azimuth_range[0], azimuth_range[1])

            # radii
            radii = np.random.choice(distance_distribute)

            # generate sound source position based on azimuth, elevation and radii.\
            [offset_x, offset_y, offset_z] = sph2cart(
                azimuth, elevation, radii)
            offset = np.array([offset_x, offset_y, offset_z])
            source = offset + micpos

            # Check if the source position is within the correct range, otherwise resample
            # When the sound source locations (target or nosie) are chosen, 
            # we assume that they are at least 0.5 meters away from the wall.
            if np.all(source <= room-space) and np.all(source >= space):
                return source

        return None
