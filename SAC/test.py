# import numpy as np
# data=np.load("pretrain_path_with_collision.npy")

# print(data)
import re
import numpy as np

# Define the string
with open('pretrain.txt', 'r') as f:
    # Read the contents of the file
    contents = f.read()
# string = '这样的数组如何用正则化表达式提取？[ 2.05207631e-01 -3.29544544e-01  1.77750003e+00  2.42492199e-01 -2.49862716e-01  1.89042628e+00  4.96664315e-01 -2.40912095e-01  2.00590801e+00  5.00246584e-01 -1.93922326e-01  2.10141778e+00  3.84217739e-01 -4.02843878e-02  2.06276679e+00  3.18144441e-01 8.41893703e-02  2.06342554e+00  3.57533693e-01 6.65066019e-02 1.97278261e+00 4.05394912e-01 1.08413771e-01 1.93765736e+00]'
# Define the regular expression pattern
pattern = r'\[([\d\s\.\-\+ e]+)\]'

# Find all matches of the pattern in the input string
matches = re.findall(pattern, contents)

# Convert each match to a numpy array
arrays = [np.fromstring(match, sep=' ') for match in matches]

paths=[]
# Print the resulting arrays
for array in arrays:
    if array.shape[0]==57:
        paths.append(array)
print(len(paths))        
assert len(paths)==5000
np.save("pretrain_path_with_collision.npy",paths)

pretrained_paths = np.load('pretrain_path_with_collision.npy')
pretrain_len=len(pretrained_paths)
print(pretrain_len)

