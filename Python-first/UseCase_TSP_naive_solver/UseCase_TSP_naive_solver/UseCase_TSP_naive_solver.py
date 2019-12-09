# Imports
from random import seed
from random import random
from math import sqrt
from datetime import datetime

seed(200)

class point2D:
    # Point in 2D space with name, X and Y
    def __init__(self,name,x=0,y=0):
        self.name=name
        self.x=x
        self.y=y

class linesegment:
    # segment consisting of 2 points with a name and a norm
    def __init__(self,name,p1,p2):
        self.name=name
        self.p1=p1
        self.p2=p2
        self.calculate_norm()
    
    def calculate_norm(self):
        self.norm = sqrt((self.p1.x - self.p2.x)**2 + (self.p1.y - self.p2.y)**2 )

class polygonchain:
    # polygonchain consisting of n segments with a name and a norm
    def __init__(self,name):
        self.name = name
        self.points=[]
        self.nbpoints=0
        self.seglist = []
        self.length = 0

    def add_segment(self,segment):
        self.seglist.append(segment)
        self.length += segment.norm

    def add_point(self,point):
        self.points.append(point)
        self.nbpoints += 1
        self.name += point.name + "-"

# Script functions:

# return a list containing n point2D 
# named increasingly for example "P" + n
# with x and y generated randomly within [0;spacesize]
def generate_random_points(n=50):
    spacesize=10
    pointlist=[]
    for i in range(0,n):
        point = point2D(name="P"+str(i),x=spacesize*random(),y=spacesize*random())
        pointlist.append(point)
    return pointlist

# return a list containing all possible linesegment (n!)
def build_all_segments(point_list):
    segmentlist=[]
    for idx,startpoint in enumerate(point_list):
        for endpoint in point_list[idx+1:]:
            segname = startpoint.name + "-" + endpoint.name
            segment = linesegment(segname,startpoint,endpoint)
            segmentlist.append(segment)
    return segmentlist

# return a list of polygonchain
def build_all_polygon_chain(point_list):
    
    polychainlist=[]
    
    # Always start loop at point 0
    startpoint = point_list[0]
    pc0 = polygonchain("First")
    pc0.add_point(startpoint)
    polychainlist.append(pc0)
    
    # for each level
    for n in range(0,len(point_list)-1):
        
        # take the exisiting polygonalchains
        oldpolychainlist=polychainlist[:]
        for oldpc in oldpolychainlist:
            # add points that were not previously in the chain
            remaining_point_list = point_list[:]
            pc_plist=[]
            for point in oldpc.points:
                pc_plist.append(point)
                remaining_point_list.remove(point)
            # Create new chains with +1 point
            for newpoint in remaining_point_list:
                pc = polygonchain("")
                for oldpoint in pc_plist:
                    pc.add_point(oldpoint)
                    lastpoint=oldpoint
                pc.add_point(newpoint)
                pc.length = oldpc.length + sqrt((lastpoint.x - newpoint.x)**2 + (lastpoint.y - newpoint.y)**2 )
                polychainlist.append(pc)
            # remove chain with not enough point
            polychainlist.remove(oldpc)

    # Close the Loop
    for pc in polychainlist:

        pc.add_point(startpoint)
        lastpoint = pc.points[-1]
        pc.length += sqrt((lastpoint.x - startpoint.x)**2 + (lastpoint.y - startpoint.y)**2 )

    return polychainlist


# Main:

t1 = datetime.now()

nb_points= 10
plist = generate_random_points(nb_points)
#slist = build_all_segments(plist)
pclist = build_all_polygon_chain(plist)

for p in plist:
    print(p.name + ": ({};{})".format(p.x,p.y))

#for s in slist:
#    print(s.name + ": {}".format(s.norm))

smallest_length = pclist[0].length
smallest_name = pclist[0].name
for pc in pclist:
    #print(pc.name + ":" + str(pc.length))
    if pc.length < smallest_length:
        smallest_length = pc.length
        smallest_name = pc.name


t2 = datetime.now()
duration=t2-t1
print("computation time :{}".format(duration))
print("Shorest path is " + smallest_name + ": {}".format(smallest_length))
