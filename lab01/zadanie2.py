import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
matplotlib.use('TkAgg')

I = plt.imread('mandril.jpg')

plt.figure(1)  # create figure
plt.imshow(I)  # add image
plt.title('Mandril')  # add title
plt.axis('off')  # disable display of the coordinate system
x = [ 100, 150, 200, 250]
y = [ 50, 100, 150, 200]
plt.plot(x,y,'r.',markersize=10)
plt.show()  # display
# plt.imsave('mandril2.png',I)


