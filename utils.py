import numpy as np

def get_total_point_y_3d(total_point_y):
    temp = []
    for i in range(len(total_point_y)):
        temp.append([total_point_y[i]])
    temp2 = np.array(temp)
    total_point_y_3d = np.expand_dims(temp2, axis=0)
    return total_point_y_3d
