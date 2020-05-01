from PIL import Image  # https://pillow.readthedocs.io/en/stable/handbook/index.html
import numpy as np
import os

if __name__ == '__main__':
    root_path = r'C:\Users\Sarah\Documents\Python\Bax-analysis\IF36_selected-for-analysis-with-Jan'
    output_path = os.path.join(root_path, 'test.tiff')

    array = np.array(range(256*256))
    array = array.reshape((256,256))

    img = Image.fromarray(array)
    print(img)
    img.save(output_path, format='tiff')
