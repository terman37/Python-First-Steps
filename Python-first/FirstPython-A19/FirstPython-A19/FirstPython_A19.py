

# sequential execution - function should be before call
def myfunc():
    pass

# a,b,c and d are somehow global variables
# they have an environment: the whole interpreter itself
a=10
b=20
c=10
d = a+b+c

# id gives the object identifier - RAM address of pyobject structure
print("Id of A:" + str(id(a)))
print(d)