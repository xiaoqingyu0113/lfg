from lfg.ros import LFG, DTYPE
from lfg.derive import predict
import matplotlib.pyplot as plt
import numpy as np








fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


p0 = np.array([0,0,3]).astype(DTYPE)
v0 = np.array([5,0,3]).astype(DTYPE)
w0 = np.array([1,0,0]).astype(DTYPE)


predicted_points = predict(p0, v0, w0, 3.0, 3000)
ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='g') # predicted trajectory

predicted_points = predict(p0, v0, w0, 3.0, 1000)
ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='g') # predicted trajectory


predicted_points = predict(p0, v0, w0, 3.0, 600)
ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='g') # predicted trajectory


predicted_points = predict(p0, v0, w0, 3.0, 300)
ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='g') # predicted trajectory

predicted_points = predict(p0, v0, w0, 3.0, 100)
ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='b') # predicted trajectory


predicted_points = predict(p0, v0, w0, 3.0, 50)
ax.plot(predicted_points[:,0], predicted_points[:,1], predicted_points[:,2], color='y') # predicted trajectory

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-10, 20)
ax.set_ylim(-15, 15)
ax.set_zlim(-10, 15)

plt.show()




