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
        rstr = self.name + ": ( %.3f ; %.3f )" % (self.x, self.y)
        return rstr

class polygonchain:
    # polygonchain consisting of n points2D with a name and length
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

    def __lt__(self, other):
        return self.length < other.length
    
    def __gt__(self,other):
        return self.length > other.length

    def __str__(self):
        rstr = "Path : " + self.name + " ; Length = %.3f" % self.length
        return rstr

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

    
    # Create initial polygonnchain starting at point 0
    startpoint = point_list[0]
    pc = polygonchain("initial polygonnchain")
    pc.add_point(startpoint)
    # Initialize polygonchain list
    polychainlist = []
    polychainlist.append(pc)

    # For each level (except 0 as it's starting point)
    for n in range(1, len(point_list)):
        # Take the exisiting polygonalchains
        polychainlist_copy = polychainlist[:]
        for incomplete_polychain in polychainlist_copy:
            # Split point list between points that were/were not already in polygonchain
            remaining_point_list = point_list[:]
            points_already_added = []
            for point in incomplete_polychain.points:
                points_already_added.append(point)
                remaining_point_list.remove(point)
            # Create new chain with +1 point
            for newpoint in remaining_point_list:
                pc = polygonchain("")
                for point in points_already_added:
                    pc.add_point(point)
                pc.add_point(newpoint)
                polychainlist.append(pc)
            # Remove incomplete chain from the list
            polychainlist.remove(incomplete_polychain)

    # Close the Loop by going back at start point (point0)
    for pc in polychainlist:
        pc.add_point(startpoint)

    return polychainlist


def main():
    t1 = datetime.now()

    nb_points = 9
    point_list = generate_random_points(nb_points)
    polygonchain_list = build_all_polygon_chain(point_list)

    # Display point coordinates and add it on plot
    print("\nGenerated points:")
    for p in point_list:
        print(p)
        plt.scatter(p.x, p.y)
        plt.text(p.x + .1, p.y + .1, p.name, fontsize=10)

    # Find polygonal chain with smallest length
    shortest_path = min(polygonchain_list)
    print("\n--> Shortest %s" % shortest_path)

    # Longest path (to compare...)
    longest_path = max(polygonchain_list)
    print("\nFor info:\n- Worst %s" % longest_path)
    print("- Calculated total of %d paths possible through %d points" % (len(polygonchain_list),len(point_list)))

    # Draw path on plot
    xlist = [p.x for p in shortest_path.points]
    ylist = [p.y for p in shortest_path.points]
    plt.plot(xlist, ylist, linestyle='-')

    # Display computation time (start to be very long from 10 points => 9! = 362,880 paths)
    t2 = datetime.now()
    duration = t2 - t1
    print("- Computation time :{}".format(duration))

    # Finally show plot
    plt.show()


# Main:
if __name__ == "__main__":
    main()
