#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import pprint
import sys
import argparse
import pymongo
import datetime
import matplotlib
import matplotlib.pyplot as plt
import itertools
import math

parser = argparse.ArgumentParser(description='2D-scatter-plot.')
parser.add_argument('--mongodbport', dest='mongodbport', action='store', default=None, help='Port of MongoDB', required=True)
parser.add_argument('--mongodbip', dest='mongodbip', action='store', default=None, help='IP of MongoDB', required=True)
parser.add_argument('--project', dest='project', action='store', default=None, help='Project name', required=True)

args = parser.parse_args()

def dier (msg):
    pprint.pprint(msg)
    sys.exit(1)

#dier(args)
#dier(args.mongodbip)

def nearest_non_prime(n):
    while is_prime(n):
        n = n + 1
    return n

def is_prime(n):
    if n == 1:
        return False
    if n % 2 == 0 and n > 2:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
        return True
    return True

def get_axis_label (param_names, x):
    x = x - 1
    return param_names[x]
    return "TODO CHANGE ME"

def get_index_of_minimum_value(array):
    minimum_index = None
    minimum = float('inf')
    index = 0
    for item in array:
        if item < minimum:
            minimum = item
            minimum_index = index
        index = index + 1
    return minimum_index

def get_index_of_maximum_value(array):
    maximum_index = None
    maximum = float("-inf")
    index = 0
    for item in array:
        if item > maximum:
            maximum = item
            maximum_index = index
        index = index + 1
    return maximum_index

def get_min_value(array):
    minimum = float('inf')
    for item in array:
        if item < minimum:
            minimum = item
    return minimum


def get_max_value(array):
    maximum = float('-inf')
    for item in array:
        if item > maximum:
            maximum = item
    return maximum

def findsubsets(S, m):
    data = set(itertools.combinations(S, m))
    return data

def get_largest_divisors(n):
    if n == 0 or n == 1:
        return {'x': 2, 'y': 1}

    if n == 2:
        return {'x': 2, 'y': 1}

    n = nearest_non_prime(n)

    sqrtnr = int(math.ceil(math.sqrt(n)))

    for i in reversed(range(2, sqrtnr + 1)):
        if n % i == 0:
            y = int(n / i)
            if i > y:
                return {'x': i, 'y': y}
            else:
                return {'x': y, 'y': i}
    raise Exception("Couldn't get any divisors for n = " + str(n))



def readable_time (x):
    return str(datetime.timedelta(seconds=x))

def _plotdata_singleaxis (x, loss, axarr, param_names):
    min_x_str = str(x[get_index_of_minimum_value(loss)])

    max_x_str = str(x[get_index_of_maximum_value(loss)])
    min_at_desc = " (min: " + min_x_str + ", max: " + max_x_str + ")"

    for ax in axarr.flat:
        ax.set_xlabel(get_axis_label(param_names, 1))
        ax.set_ylabel("loss")

    #x = get_parameter_name_from_value(1, x)

    plt.plot(x, loss, 'x', color='black', alpha=0.3);

def _plotdata(axarr, axis, data, param_names):
    if axis[0] == 0 or axis[1] == 0:
        raise Exception("Axis must begin at 1, not 0!")

    if not "dimensions" in data:
        raise Exception("The data must contain a subdictionary called `dimensions`")

    axis_warning = "The data must contain a subdictionary called `dimensions` with values for the axis "
    if not str(axis[0]) in data["dimensions"]:
        raise Exception(axis_warning + str(axis[0]) + " (" + get_axis_label(param_names, axis[0]) + ")")

    if not str(axis[1]) in data["dimensions"]:
        raise Exception(axis_warning + str(axis[1]) + " (" + get_axis_label(param_names,  axis[1]) + ")")

    if not len(axis) == 2:
        raise Exception("Can only plot 2 dimensions, not more or less!")

    data1 = data["dimensions"][str(axis[0])]
    data2 = data["dimensions"][str(axis[1])]
    heatmap = data["loss"]

    if len(data1) != len(data2):
        raise Exception("Both axis must contain the same number of data!")

    if len(data1) != len(heatmap):
        raise Exception("All axes must contain the same number of data as the parameter-array!")

    x = data["dimensions"][str(axis[0])]
    y = data["dimensions"][str(axis[1])]
    e = data["loss"]
    axis = [int(axis[0]), int(axis[1])]

    string_desc = get_axis_label(param_names, axis[0]) + ' - ' + get_axis_label(param_names, axis[1])

    index_of_min_value = get_index_of_minimum_value(e)

    min_x_str = str(x[index_of_min_value])
    min_y_str = str(y[index_of_min_value])

    index_of_max_value = get_index_of_maximum_value(e)
    max_x_str = str(x[index_of_max_value])
    max_y_str = str(y[index_of_max_value])
    min_at_desc = " (min: " + min_x_str + ", " + min_y_str + ", max: " + max_x_str + ", " + max_y_str + ")"

    size_in_px = 7

    sc = axarr.scatter(x, y, c=e, s=size_in_px, cmap='jet', edgecolors="none")
    plt.colorbar(sc, ax=axarr)
    axarr.grid()
    axarr.autoscale(enable=True, axis='both', tight=True)
    axarr.relim()
    axarr.autoscale_view()

    stretch_factor = 0.1

    min_x = get_min_value(x)
    max_x = get_max_value(x)
    width = max_x - min_x
    additional_width = width * stretch_factor

    min_y = get_min_value(y)
    max_y = get_max_value(y)
    height = max_y - min_y
    additional_height = height * stretch_factor

    axarr.set_xlim(min_x - additional_width, max_x + additional_width);
    axarr.set_ylim(min_y - additional_height, max_y + additional_height);

    axarr.set_xlabel(get_axis_label(param_names, axis[0]))
    axarr.set_ylabel(get_axis_label(param_names, axis[1]))
    axarr.set_title(string_desc + ', loss: ' + min_at_desc, fontsize=9)

def plotdata(data):
    keys = list(data["dimensions"].keys())

    permutations = findsubsets(keys, 2)
    number_of_permutations = len(permutations)

    layout = get_largest_divisors(number_of_permutations)
    if layout is None:
        raise Exception("ERROR getting the layout: get_largest_divisors(" + str(number_of_permutations) + ") = " + str(layout))

    if len(keys) == 1 or len(keys) == 2:
        layout["x"] = 1
        layout["y"] = 1
    else:
        if layout["x"] == 1:
            layout["x"] = 2

        if layout["y"] == 1:
            layout["y"] = 2

    f, axarr = plt.subplots(layout["x"], layout["y"], squeeze=False)
    f.set_size_inches(20, 15, forward=True)
    plt.subplots_adjust(wspace=0.15, hspace=0.75)

    permutations_array = []
    for item in sorted(permutations):
        permutations_array.append(item)

    i = 0
    if len(keys) == 1:
        _plotdata_singleaxis(data["dimensions"]["1"], data["loss"], axarr, data["param_names"])
    else:
        for row in axarr:
            for col in row:
                if i < len(permutations_array):
                    _plotdata(col, permutations_array[i % len(permutations_array)], data, data["param_names"])
                else:
                    col.axis('off')
                i = i + 1

def main():
    connect_string = "mongodb://" + str(args.mongodbip) + ":" + str(args.mongodbport) + "/"

    client = pymongo.MongoClient(connect_string)
    db = client[args.project]
    jobs = db["jobs"]

    values = []
    for x in jobs.find({"result.status": "ok"}, {"result.loss": 1, "misc.vals": 1, "misc.vals": 1}):
        values.append(x)               

    client.close()

    param_names = []
    e = []
    dimensions = {}

    number_of_values = 0

    best_value_data = None
    best_value = float("inf")

    for x in values:
        value = float('inf')

        value = x["result"]["loss"]

        if value < best_value:
            best_value_data = x
        e.append(float(value))

        i = 1
        vals = x["misc"]["vals"]
        for val in sorted(vals):
            if val not in param_names:
                param_names.append(val)
            thisval = vals[val][0]
            if str(i) not in dimensions:
                dimensions[str(i)] = []
            temp = dimensions[str(i)]
            temp.append(thisval)
            i = i + 1
        number_of_values = number_of_values + 1

    data = {"dimensions": dimensions, "loss": e, "param_names": param_names}

    title = "f(x_0, x_1, x_2, ...) = loss, min at f(";

    title = 'Best value not known!'
    if best_value_data is not None:
        title = args.project + '('
        var_for_title = []
        x_number = 1
        for item in sorted(best_value_data["misc"]["vals"]):
            this_value = best_value_data["misc"]["vals"][item][0]
            var_for_title.append(get_axis_label(param_names, x_number) + " = " + str(this_value))
            x_number = x_number + 1
        title = title + ', '.join(var_for_title) + ") = " + str(best_value_data["result"]["loss"])
        title = title + "\nProject: " + str(args.project)
        title = title + ", Number of evals: " + str(number_of_values) + ", " + "Number of dimensions: " + str(len(dimensions.keys()))

    plotdata(data)

    plt.suptitle(title)

    plt.show()

    plt.close('all')

main()
