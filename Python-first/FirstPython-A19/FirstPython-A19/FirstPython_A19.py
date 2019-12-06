

# sequential execution - function should be before call
def myfunc():
    pass #means you do nothing

def factorial(n):
    result = 1
    print(type(n))
    if n>2:
        for i in range(n,1,-1):
            result *= i
    elif n<0:
        raise Exception("The value of {} cannot be raised to the factorial".format(n))
    return result


try:
    print(factorial("A"))
except:
    print("Cannot compute factorial !")


class Point2D:
    #only one constructor, no overload
    def __init__(self,x=None,y=None):
        self.x=x
        self.y=y
    
    def __str__(self):
        mystr=("im point2D instance, my coordinates are ({};{})".format(self.x,self.y))   
        return mystr

p1=Point2D(y=5)
p1.x=50
print(p1)