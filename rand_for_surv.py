from random import randrange, choice, shuffle
from os import listdir


# generate n melodies to put on survey, and a random order
def main():
    n = 5
    for i in range(1, 7):
        print(f"run {i}:")
        for j in range(n):
            print(f"melody: {randrange(1, 101)}")
        print()

    print("title:")
    files = listdir("melodies/title")
    for i in range(n):
        print(f"melody: {choice(files)}")
    print()

    print("battle")
    files = listdir("melodies/battle")
    for i in range(n):
        print(f"melody: {choice(files)}")

    melodies = list(range(1, n*8 + 1))
    shuffle(melodies)
    for i in range(len(melodies)):
        print(melodies.pop())
