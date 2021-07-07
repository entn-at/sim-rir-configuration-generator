#!/usr/bin/python
# -*- coding: utf-8 -*-

import param_generator

if __name__ == '__main__':
    room_param_generator = param_generator.ParameterGenerator(
        sample_num=3,
        seed=42,
        config={
            'room': {
                'x': [3, 10],
                'y': [3, 8],
                'z': [2.5, 6]
            },
            'target': {
                'azimuth': [-180, 180],
                'elevation': [-30, 180]
            },
            'noise': {
                'number': 3,
                'azimuth': [-180, 180],
                'elevation': [-30, 180]
            },
        }
    )

    for config in room_param_generator:
        if config is None:
            continue

        for source_type, config in config.items():
            print("="*20, source_type, "="*20)
            if source_type == "noise_sources":
                for i in config:
                    print("    ", i)
            else:
                print("    ", config)
