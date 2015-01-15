from __future__ import division
from Tkinter import Tk
from tkFileDialog import askopenfilename
from PIL import Image
from PIL import ImageDraw
from copy import deepcopy
import pickle
import itertools
import random
import numpy as np
import gc
import collections
import functools

POLY_COUNT = 371
RGBA_GAUSS_VALUE = 25.6
COORD_GAUSS_MULT = .1
GEN_OUTPUT_STEP = 500
GEN_CHANGE_THRESHOLD = 0.25
GC_STEP = 1000

root = Tk()
root.withdraw()
orig_image = Image.open(askopenfilename()).convert("RGB")
orig_data = orig_image.load()
width = orig_image.size[0]
height = orig_image.size[1]

flatten = lambda l: list(itertools.chain(*l))
NP_INT16 = lambda image: np.array(flatten(list(image.getdata())),np.int16)
orig_np = NP_INT16(orig_image)

def get_delta(image):
    image_np = NP_INT16(image)
    return abs(orig_np - image_np).sum() / (width * height)

poly_cache = {}
GET_POLY_CODE = lambda polygon: ''.join(map(str,polygon.values())).replace(' ','')
global blankimage
blankimage = Image.new('RGB',[width,height])
for x,y in itertools.product(*tuple(map(range,(width,height)))):
    blankimage.putpixel((x,y),(0,0,0))
blankimage = blankimage.getdata()

def create_image(polygons):
    global blankimage
    image = Image.new('RGB',[width,height])
    image.putdata(blankimage)
    for polygon in polygons:
        poly_code = GET_POLY_CODE(polygon)
        if poly_code in poly_cache:
            image.paste(poly_cache[poly_code][0], (0,0), poly_cache[poly_code][1])
        else:
            lines = map(tuple,polygon['coords'])
            layer = Image.new('RGB',[width,height])
            mask = Image.new('L',[width,height])
            layer_draw = ImageDraw.Draw(layer)
            layer_draw.polygon(lines,fill=tuple(polygon['rgb']))
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(lines,fill=round(polygon['alpha']))
            image.paste(layer, (0,0), mask)
            poly_cache[poly_code] = (layer,mask)
    return image

RGB_GAUSS = lambda x: random.gauss(x,RGBA_GAUSS_VALUE)
VAL_RESTRICT = lambda high: lambda x: int(min(max(round(x),0),high))
RGB_RESTRICT = VAL_RESTRICT(255)
RAND_INC_DEC = lambda y: random.choice(((lambda x: x+1), (lambda x: x-1)))(y)

def mutate(polygons,gauss):
    global last_mutation
    null_alpha = None
    for polygon in polygons:
        if polygon['alpha'] == 0:
            null_alpha = polygon
            break
    if null_alpha:
        polygons.remove(null_alpha)
        polygons.append(random_polygon())
        last_mutation = "new"
        print "*\b\b",
        return polygons
    index = random.choice(range(len(polygons)))
    polygon = polygons[index]
    globals()['last_mod_polygon_index'] = index
    polygons.remove(polygon)
    choice = random.choice(['rgb','alpha','coords'])
    if gauss == 'gaussian':
        if choice == 'rgb':
            value = polygon[choice]
            subindex = random.choice(range(len(value)))
            value[subindex] = RGB_GAUSS(value[subindex])
            value[subindex] = RGB_RESTRICT(value[subindex])
        elif choice == 'alpha':
            value = RGB_RESTRICT(RGB_GAUSS(polygon[choice]))
        else:
            value = polygon[choice]
            subindex = random.choice(range(len(value)))
            value[subindex] = (lambda x: [random.gauss(x[0],COORD_GAUSS_MULT*width),\
                                   random.gauss(x[1],COORD_GAUSS_MULT*height)])(value[subindex])
            value[subindex] = (lambda x: [VAL_RESTRICT(width)(x[0]),VAL_RESTRICT(height)(x[1])])(value[subindex])
    elif gauss == 'random':
        if choice == 'rgb':
            value = map(lambda x: random.randrange(0,255),list(polygon[choice]))
        elif choice == 'alpha':
            value = random.randrange(0,255)
        else:
            value = map(lambda x: [random.randrange(0,width),\
                                   random.randrange(0,height)], polygon[choice])
    elif gauss == 'removal':
        choice = 'alpha'
        value = 0
    else:
        if choice == 'rgb':
            value = polygon[choice]
            subindex = random.choice(range(len(value)))
            value[subindex] = (value[subindex])
            value[subindex] = RGB_RESTRICT(value[subindex])
        elif choice == 'alpha':
            value = RGB_RESTRICT(RAND_INC_DEC(polygon[choice]))
        else:
            value = polygon[choice]
            subindex = random.choice(range(len(value)))
            value[subindex] = random.choice(((lambda x: [RAND_INC_DEC(x[0]), x[1]]),\
                                             (lambda x: [x[0], RAND_INC_DEC(x[1])])))\
                                             (value[subindex])
            value[subindex] = (lambda x: [VAL_RESTRICT(width)(x[0]),VAL_RESTRICT(height)(x[1])])(value[subindex])
    polygon[choice] = value
    polygons.insert(index,polygon)
    last_mutation = choice
    return polygons

def random_polygon():
    return {'rgb':map(lambda x: random.randint(0,255), range(3)),
            'alpha':random.randint(0,255),
            'coords':map(lambda x: [random.randint(0,width),random.randint(0,height)], range(3))
            }

def gen_parent():
    parent = list()
    for x in range(POLY_COUNT):
        parent.append(random_polygon())
    return parent

class Progress():
    def __init__(self):
        self.symbols = '|/-\\'
        self.pos = 0
        self.mod = len(self.symbols)
    def __call__(self):
        print self.symbols[self.pos] + '\b\b',
        self.pos = (self.pos + 1) % self.mod
        return True

def mainloop(parent=None,generations=-1):
    global debug_parent
    if parent==None:
        parent = gen_parent()
    debug_parent = parent
    i = generations
    delta = lambda x: get_delta(create_image(x))
    delta_count = 0
    parent_delta = delta(parent)
    last_gen_delta = 0
    gauss_cycle = itertools.cycle(['random','gaussian','increment'])
    gaussian = gauss_cycle.next()
    progress = Progress()
    print "Initial delta:", parent_delta
    while progress():
        i+=1
        child = mutate(deepcopy(parent),gaussian)
        child_delta = delta(child)
        if child_delta == parent_delta:
            parent = child
            parent_delta = child_delta
            debug_parent = parent
            #print '\tNo change. (%s, %s)'%(last_mutation,str(i))
        if child_delta < parent_delta:
            print "\t", parent_delta - child_delta, "(%s, %s)"%(last_mutation,str(i)),
            delta_count += 1
            parent = child
            parent_delta = child_delta
            debug_parent = parent
            try:
                f = open("jenny.algo.raw",'w')
                f.write(str(parent))
                f.close()
                print ''
            except:
                print ", but couldn't output."
        if i % GEN_OUTPUT_STEP == 0: #Output information, picture
            f = open("generations",'w')
            f.write(str(i))
            f.close()
            create_image(parent).save(str(i) + ".jpeg","JPEG")
            print "Gen.",str(i),":: Delta:",parent_delta,"Change:",str(parent_delta - last_gen_delta),'/',delta_count
            delta_count = 0
            if last_gen_delta - parent_delta < GEN_CHANGE_THRESHOLD:
                gaussian = gauss_cycle.next()
                print " Mode change: " + gaussian
            last_gen_delta = parent_delta
        if i % GC_STEP == 0: #clean up poly image cache
            poly_codes = map(GET_POLY_CODE,parent)
            for poly_code in poly_cache.keys():
                if poly_code not in poly_codes:
                    poly_cache.pop(poly_code)

try:
    try:
        f = open('jenny.algo.raw','r')
        contents = f.read()
        f.close()
        parent = eval(contents.replace('.0',''))
        f = open('generations','r')
        contents = f.read()
        f.close()
        generations = eval(contents.replace('.0',''))
        raw_input("GEN: " + str(generations) + ". Approve?")
        mainloop(parent,generations)
    except IOError:
        mainloop()
except Exception as e:
    print type(e), e
    raw_input()


# New code starts here:


# General functions:

def compose(*fns):
    return reduce(lambda x, y: lambda z: x(y(z)), fns)


def int_clip(max_, val):
    return int(min(max(round(val), 0), max_))


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
    for i in itertools.count(time + 1):
        current_temperature = temperature(i)
        new_solution = mutate(solution)
        new_fitness = fitness(new_solution)
        if new_fitness > current_fitness or current_temperature > random.random():
            current_fitness = new_fitness
            solution = new_solution
        if stop(time, current_temperature, current_fitness):
            return i, solution
