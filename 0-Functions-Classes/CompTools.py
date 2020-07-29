import os

#type hint for input parameters for python >3.5
def multiplyList(aList : list, n):
    j = 0
    for i in aList:
        aList[j] = i * n
        j += 1
    return aList

# lambda programming
# lbd essentially a function pointer
def transform_to_newlist(aList: list,lbd):
    # apply lbd operator to each element of the list
    return [lbd(x) for x in aList]

def apply_to_list(aList: list,lbd):
    # apply lbd operator to each element of the list
    j = 0
    for i in aList:
        aList[j] = lbd(i)
        j += 1
    return aList

def apply_to_list_v2(aList: list,lbd):
    map(lbd,aList)

def genSquares(n=10):
    print("Generating squares of numbers from 1 to {}".format(n))
    for i in range(1,n+1):
        yield i**2

def emit_lines(pattern="DSTI", originPath = "c:\temp"):
    lines = []
    #browses the directory called test in the current directory of execution
    for dir_path, dir_names, file_names in os.walk(originPath):
        #iterate over the all the files in test
        for file_name in file_names:
            if file_name.endswith('.txt'):
                #if current file has a .txt extension
                for line in open(os.path.join(dir_path, file_name)):
                    #open the file and load its content in memory, by providing the full file path
                    #open provide a collection of lines in the file
                    if pattern in line:
                        #if pattern found in line, add the line in the lines list
                        lines.append(line)
    return lines

#now, let's divide this code into various functions yielding generators

def generate_filenames(directory="./"):
    """
    generates a sequence of opened files
    matching a specific extension
    """
    for dir_path, dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            if file_name.endswith('.txt'):
                yield open(os.path.join(dir_path, file_name))

def cat_files(files):
    """
    takes in an iterable of filenames
    """
    for fname in files:
        for line in fname:
            yield line

def grep_files(lines, pattern=None):
    """
    takes in an iterable of lines
    """
    for line in lines:
        if pattern in line:
            yield line


class myGoodTools:
    #syntaxically speaking, a method is "static" in Python if there's no "self" parameter

    #attribute
    #class attribute, shared by all instances of the class (static attribute in C++ / Java, etc.)
    #in C++ and others, only a static method can change the value of a static attribute
    #class methods don't exists in C++ etc. --> static methods
    counter = 50

    #making matters clear with decorators
    @staticmethod
    def staticmethod():
        myGoodTools.counter=500
        return "I am a static method"

    @classmethod
    def classmethod(cls): #cls is the name of the class
        cls.counter = cls.counter+1
        return "I am a class method: {}".format(cls.counter), cls

    #instance method, no particular decorator
    def method(self):
        return "I am an instance method {}".format(self.counter), self