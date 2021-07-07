#!/usr/bin/python
# -*- coding: utf-8 -*-

import param_generator

import numpy as np

if __name__ == '__main__':
    pa = param_generator.ParameterGenerator(
        sample_num=3,
        seed=1234,
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

    for p in pa:
        if p is None:
            continue

        for k, v in p.items():
            print("="*20, k, "="*20)
            if k == "noise_sources":
                for v in v:
                    print("    ", v)
            else:
                print("    ", v)
