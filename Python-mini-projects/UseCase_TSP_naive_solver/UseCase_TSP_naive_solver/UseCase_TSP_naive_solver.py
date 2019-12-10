# Imports
from random import seed
from random import random
from math import sqrt
from datetime import datetime
import matplotlib.pyplot as plt


# seed(200)

class point2D:
    # Point in 2D space with name, X and Y
    def __init__(self, name, x=0, y=0):
        self.name = name
        self.x = x
        self.y = y

    def __str__(self):
        return (self.name + ": ({};{})".format(self.x, self.y))


class polygonchain:
    # polygonchain consisting of n points with a name and length
    def __init__(self, name):
        self.name = name
        self.points = []
        self.nbpoints = 0
        self.length = 0

    def add_point(self, point):
        if self.nbpoints != 0:
            lastpoint = self.points[-1]
            self.add_length(lastpoint, point)
        self.points.append(point)
        self.nbpoints += 1
        self.name += point.name + "-"

    def add_length(self, lastpoint, newpoint):
        self.length = self.length + sqrt((lastpoint.x - newpoint.x) ** 2 + (lastpoint.y - newpoint.y) ** 2)


def generate_random_points(n=50, spacesize=10):
    """ 
    return a list containing n point2D 
    named increasingly for example "P" + n
    with x and y generated randomly within [0;spacesize]
    """

    pointlist = []
    for i in range(0, n):
        point = point2D(name="P" + str(i), x=spacesize * random(), y=spacesize * random())
        pointlist.append(point)
    return pointlist


def build_all_polygon_chain(point_list):
    """ 
    generates a list containing all possible polygonal chains 
    from a given list of points
    """

    polychainlist = []

    # Always start loop at point 0
    startpoint = point_list[0]
    pc = polygonchain("First chain")
    pc.add_point(startpoint)
    polychainlist.append(pc)

    # for each level (except 0 as starting point)
    for n in range(1, len(point_list)):
        # take the exisiting polygonalchains
        oldpolychainlist = polychainlist[:]
        for oldpc in oldpolychainlist:
            # add points that were not previously in the chain
            remaining_point_list = point_list[:]
            pc_plist = []
            for point in oldpc.points:
                pc_plist.append(point)
                remaining_point_list.remove(point)
            # Create new chains with +1 point
            for newpoint in remaining_point_list:
                pc = polygonchain("")
                for oldpoint in pc_plist:
                    pc.add_point(oldpoint)
                pc.add_point(newpoint)
                polychainlist.append(pc)
            # remove chain with not enough point
            polychainlist.remove(oldpc)

    # Close the Loop
    for pc in polychainlist:
        pc.add_point(startpoint)

    return polychainlist


def main():
    t1 = datetime.now()

    nb_points = 8
    plist = generate_random_points(nb_points)
    pclist = build_all_polygon_chain(plist)

    # display point coordinates
    for p in plist:
        print(p)
        plt.scatter(p.x, p.y)
        plt.text(p.x + .1, p.y + .1, p.name, fontsize=10)

    # find shortest path
    shortest_path = pclist[0]
    for pc in pclist:
        # print("Path: {}, length: {}".format(pc.name,pc.length))
        if pc.length < shortest_path.length:
            shortest_path = pc

    xlist = [p.x for p in shortest_path.points]
    ylist = [p.y for p in shortest_path.points]
    plt.plot(xlist, ylist, linestyle='-')

    print("Shortest path is " + shortest_path.name + ": {}".format(shortest_path.length))

    # Display computation time (start to be very long from n=10)
    t2 = datetime.now()
    duration = t2 - t1
    print("computation time :{}".format(duration))

    plt.show()


# Main:
if __name__ == "__main__":
    main()
