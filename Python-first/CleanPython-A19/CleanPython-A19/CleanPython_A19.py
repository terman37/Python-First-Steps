import module1

print('__name__ value : ' + __name__)

def main():
    print('__name__ value : ' + __name__)
    print('This is my main function')


if __name__ == "__main__":
    main()
    module1.test()
