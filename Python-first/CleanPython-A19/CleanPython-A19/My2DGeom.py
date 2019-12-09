
class Point2D:
    #only one constructor, no overload
    def __init__(self,x=None,y=None):
        self.x=x
        self.y=y
    
    def __str__(self):
        mystr=("im point2D instance, my coordinates are ({};{})".format(self.x,self.y))   
        return mystr

def testfunc():
    print('__name__ value : ' + __name__)
    
