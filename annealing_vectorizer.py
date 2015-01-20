from __future__ import division

import itertools
import random
import collections
import functools
import math
import json
import numpy as np

from PIL import Image, ImageDraw
from Tkinter import Tk
from tkFileDialog import askopenfilename


# General functions:

def compose(*fns):
    return reduce(lambda x, y: lambda z: x(y(z)), fns)


def int_clip(max_, val):
    return int(min(max(round(val), 0), max_))


def cached(f):
    cache = {}
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.iteritems()))
        result = cache.get(key, None)
        if result is None:
            result = f(*args, **kwargs)
            cache[key] = result
        return result
    return wrapper


# Solutions:

Polygon = collections.namedtuple('Polygon', 'rgb alpha coordinates')

def new_polygon(width, height):
    return Polygon(
        rgb=tuple(random.randint(0, 255) for __ in xrange(3)),
        alpha=random.randint(0, 255),
        coordinates=tuple(
            (random.randint(0, width), random.randint(0, height))
            for __ in 
            xrange(3)
        )
    )


def new_solution(num_polygons, width, height):
    return tuple(new_polygon(width, height) for __ in xrange(num_polygons))


def mutate_factory(mutation_probability, gauss_sigma, width, height):
    gauss_factory = lambda max_: lambda x: random.gauss(x, gauss_sigma * max_)
    value_mutate_factory = lambda max_: compose(
        functools.partial(int_clip, max_), 
        gauss_factory(max_)
    )

    rgb_mutate = value_mutate_factory(255)
    width_mutate = value_mutate_factory(width)
    height_mutate = value_mutate_factory(height)

    def mutate():
        return mutation_probability > random.random()

    def mutate_polygon(polygon):
        rgb = tuple(
            rgb_mutate(value) if mutate() else value
            for value in
            polygon.rgb
        )
        alpha = rgb_mutate(polygon.alpha) if mutate() else polygon.alpha
        coordinates = tuple(
            (
                width_mutate(width) if mutate() else width,
                height_mutate(height) if mutate() else height
            )
            for width, height in
            polygon.coordinates
        )
        return Polygon(rgb, alpha, coordinates)

    def mutate_solution(solution):
        return tuple(
            mutate_polygon(polygon)
            for polygon in
            solution
        )

    return mutate_solution


# Simulated annealing:

def anneal(solution, fitness, mutate, temperature, time, stop):
    current_fitness = fitness(solution)
    counter = itertools.count(time + 1)
    for i in counter:
        current_temperature = temperature(i)
        new_solution = mutate(solution)
        new_fitness = fitness(new_solution)
        accepted = (
            new_fitness > current_fitness or 
            math.exp(
                (current_fitness - new_fitness) / current_temperature
            ) > random.random()
        )
        if accepted:
            current_fitness = new_fitness
            solution = new_solution
        if stop(i, current_temperature, current_fitness):
            return i, solution


def temperature_factory(sigma=4, initial=1.5):
    k = sum(9 * .1 ** (i + 1) for i in xrange(sigma))
    def temperature(time):
        return initial * k ** time
    return temperature


# Objective function

def create_image_factory(width, height):
    def new_image(mode, *args, **kwargs):
        return Image.new(mode, (width, height), *args, **kwargs)

    @cached
    def create_layer(polygon):
        layer = new_image('RGB')
        draw = ImageDraw.Draw(layer)
        draw.polygon(polygon.coordinates, fill=polygon.rgb)
        return layer

    @cached
    def create_mask(polygon):
        mask = new_image('L')
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon.coordinates, fill=polygon.alpha)
        return mask

    def create_image(polygons):
        image = new_image('RGB', 'white')
        for polygon in polygons:
            image.paste(create_layer(polygon), (0, 0), create_mask(polygon))
        return image

    return create_image


def fitness_factory(create_image, target):
    target_data = np.array([pixel for row in target.getdata() for pixel in row], np.int16)
    width, height = target.size
    area = float(width * height)

    def fitness(polygons):
        image = create_image(polygons)
        image_data = np.array([color for row in image.getdata() for color in row], np.int16)
        return ((target_data - image_data) ** 2).mean() / area

    return fitness


# Main loop

def main(target, generations=float('inf'), polygons=100, mutation_probability=0.01, gauss_sigma=0.2, step=100):
    solution = new_solution(polygons, *target.size)
    create_image = create_image_factory(*target.size)
    fitness = fitness_factory(create_image, target)
    mutate = mutate_factory(mutation_probability, gauss_sigma, *target.size)
    temperature = temperature_factory()  # expose params
    def stop(time, *args, **kwargs):
        _stop = not time % step
        if _stop:
            print time, args[0], args[1]
        return _stop
    time = 0  # expose init time
    while time < generations:
        time, solution = anneal(solution, fitness, mutate, temperature, time, stop)
        create_image(solution).save(str(time).zfill(7) + '.jpeg', 'JPEG')
        with open('raw.json', 'w') as f:
            json.dump(solution, f)


if __name__ == '__main__':
    import os
    print os.getcwd()
    root = Tk()
    root.withdraw()
    target = Image.open(askopenfilename()).convert("RGB")
    main(target)