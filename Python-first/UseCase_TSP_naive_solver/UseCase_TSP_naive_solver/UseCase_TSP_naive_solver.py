# Imports
from random import seed
from random import random

seed(200)

class point2D:
    # attributes name, X and Y
    def __init__(self,name,x=0,y=0):
        self.name=name
        self.x=x
        self.y=y

class linesegment:
    # attributes name + 2 points 2D
    # method get norm at creation
    pass

class polygonchain:
    # attribute name
    # total length
    pass


# Script functions:

def generate_random_points(n=50):
    # return a list containing n point2D
    # named increasingly for example "P" + n
    spacesize=10
    pointlist=[]
    for i in range(1,n):
        point = point2D(name="P"+str(i),x=spacesize*random(),y=spacesize*random())
        pointlist += point
    return pointlist

def build_all_segments(point_list):
    # return a list containing all possible linesegment (n!)
    pass

def build_all_polygon_chain(point_list):
    # return a list of polygonchain
    pass


# Main:

plist = generate_random_points(2)
