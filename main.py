import model
import numpy as np

if __name__ == "__main__":
    TOTALPIXEL = 15 # CHANGE
    SLITNUM = 4 # The number of slits each pixel has
    image_dict = {0:[[1, 1, 1],[1, 0, 1],[1, 0, 1],[1, 0, 1],[1, 1, 1]],
                  1:[[1, 1, 0],[0, 1, 0],[0, 1, 0],[0, 1, 0],[1, 1, 1]],
                  2:[[1, 1, 1],[0, 0, 1],[1, 1, 1],[1, 0, 0],[1, 1, 1]],
                  3:[[1, 1, 1],[0, 0, 1],[1, 1, 1],[0, 0, 1],[1, 1, 1]],
                  4:[[1, 0, 1],[1, 0, 1],[1, 1, 1],[0, 0, 1],[0, 0, 1]],
                  5:[[1, 1, 1],[1, 0, 0],[1, 1, 1],[0, 0, 1],[1, 1, 1]],
                  6:[[1, 1, 1],[1, 0, 0],[1, 1, 1],[1, 0, 1],[1, 1, 1]],
                  7:[[1, 1, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 0, 1]],
                  8:[[1, 1, 1],[1, 0, 1],[1, 1, 1],[1, 0, 1],[1, 1, 1]],
                  9:[[1, 1, 1],[1, 0, 1],[1, 1, 1],[0, 0, 1],[1, 1, 1]]}
    
    y_target_dict = {}
    for num in range(10):
        y_target = [0.00000001]*1001
        for i in range(501):
            if i > (num * 20) - 1 and i < 20 * (num + 1) :
                y_target[i] = 1
        y_target_dict[num] = np.array(y_target)
    
    model.train_test(TOTALPIXEL, SLITNUM, image_dict, y_target_dict)