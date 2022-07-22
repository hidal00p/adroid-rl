import matplotlib.pyplot as plt

# Init
def initPlt():
    plt.ion()

# Close
def closePlt():
    plt.ioff()
    plt.close()

# Stream
def rgbStream(rgb):
    print(len(rgb), len(rgb[0]))
    plt.imshow(rgb)