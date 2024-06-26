def error_map(image_1, image_2):
    
    import numpy as np
    error_map = np.abs(image_1 - image_2)
    
    temp = error_map**2
    euclidean_norm = np.sqrt( np.sum(temp) )
    return error_map, euclidean_norm


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    rec_1 = np.array([[2,2], [0,1]])
    rec_2 = np.array([[2,1], [1,1]])
    
    rec_error, rec_norm = error_map(rec_1, rec_2)
    
    plt.imshow(rec_1)
    plt.show()
    plt.imshow(rec_2)
    plt.show()
    plt.imshow(rec_error)
    plt.show()
    
    print(rec_norm)