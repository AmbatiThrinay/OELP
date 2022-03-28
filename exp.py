# import matplotlib.pyplot as plt
# import matplotlib.animation as anim



# def plot_cont(fun, xmax):
#     y = []
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)

#     def update(i):
#         yi = fun()
#         y.append(yi)
#         x = range(len(y))
#         ax.clear()
#         ax.plot(x, y)
#         print i, ': ', yi

#     a = anim.FuncAnimation(fig, update, frames=xmax, repeat=False)
#     plt.show()

# if __name__ == '__main__' :
#     plot_cont()

# from scipy import spatial
# import numpy as np
# from pygame.math import Vector2

# A = np.arange(10,250,50)
# B = np.full(A.shape,30)

# path = np.vstack((A,B)).T
# # print(path)
# # print(path)
# C = Vector2(60,30)
# xte = spatial.distance.cdist(path,np.array([[C.x,C.y]]))
# print(C,'\n',path,'\n',xte)

