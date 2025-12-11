import matplotlib.pyplot as plt

def plot_frontiers(grid, frontiers, goal):
    plt.imshow(grid, cmap="gray", origin="upper")
    if frontiers:
        ys, xs = zip(*frontiers)
        plt.scatter(xs, ys, s=5, c='lime')
    if goal:
        plt.scatter(goal[1], goal[0], s=60, c='red', marker='x')
    plt.title(f"{len(frontiers)} frontiers, goal={goal}")
    plt.savefig("live_frontiers.png")
    plt.close()
