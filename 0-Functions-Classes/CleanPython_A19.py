
# modules names shouldnot start with digit
from My2DGeom import *
from CompTools import *
import itertools
import os

def main():
    print('This is my main function')
    print('__name__ value : ' + __name__)
    L1 = [1,2,3,4]
    # specific function
    multiplyList(L1,3)
    print(L1)
    # lambda function
    print(apply_to_list(L1, lambda x: int(x/3)))
    print(L1)
    print(apply_to_list_v2(L1, lambda x: int(x*3)))
    print(L1)

    listofStrings = ["hello","foo","bar","metro","aaaa","baba"]
    # some have duplicate letters
    # sort list with 2 sorting keys
    # 1. highest number of duplicate letters in them
    # 2. Alphabetical order
    listofStrings.sort(key = lambda x: -(len(x)-len(set(x))))
    listofStrings.sort(key = lambda x: len(x)-len(set(x)),reverse=True)
    print(listofStrings)

    # use of yield generators
    squareseq = genSquares(5)
    #this call does not trigger the computation of the squares
    print(squareseq)
    for i in squareseq:
        # computation happens here, at every use of the generator
        print(i)

    BoyNames = ["Alan", "Adam", "Wes", "Will", "Albert", "Charles", "Chris"]
    # desired output, based on making a group of the current first letter in the string:
    # ["Alan", "Adam"]
    # ["Wes", "Will"]
    # ["Albert"]
    # ["Charles", "Chris"]

    first_letter = lambda x : x[0]
    for letter, names in itertools.groupby(BoyNames, first_letter):
        print(letter, list(names))

    # traditional monolithic code
    emit_lines("DSTI","c:\temp")

    # separated code, using generators
    py_files = generate_filenames("c:\temp")
    py_file = cat_files(py_files)
    lines = grep_files(py_file, 'DSTI')
    
    # real execution starts here, exhausting data based on generator
    for line in lines:
        print (line)

    print(myGoodTools.staticmethod())
    print(myGoodTools.classmethod())
    mygt = myGoodTools()
    print(mygt.method())

    print(myGoodTools.classmethod())

    mygt2 = myGoodTools()
    print(mygt2.method())
    print(mygt.method())

    print(mygt.staticmethod())

if __name__ == "__main__":
    main()
    testfunc()
    
