# argparse part of std package
import argparse
# pickle part of std package
import pickle

# Call from terminal
# python parse_args.py -h --> gives help about arguments
# pyhton parse_args.py --infile InfileName --outfile OutFileName

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, help='location of input file txt file')
    parser.add_argument('--outfile', type=str, help='location of output file bin format')
    args = parser.parse_args()

    if args.infile:
        path_in = args.infile
        with open(path_in, 'r') as f:
            print(f.read())

    if args.outfile:
        path_out = args.outfile
        with open(path_out, 'wb') as f:
            pickle.dump('TEST', f)
