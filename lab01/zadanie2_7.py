import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('TkAgg')

I = plt.imread('mandril.jpg')

fig,ax = plt.subplots(1)
rect = Rectangle((50,50),50,100,fill=False, ec='r')
plt.imshow(I)  # add image
plt.title('Mandril')  # add title
plt.axis('off')  # disable display of the coordinate system
ax.add_patch(rect)
plt.show()  # display
# plt.imsave('mandril2.png',I)


