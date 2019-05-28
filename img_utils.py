import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(path):

    fig, ax = plt.subplots(figsize=(8, 8))
    image = mpimg.imread(path)
    ax.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()