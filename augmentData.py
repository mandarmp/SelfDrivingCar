import numpy as np
import cv2
from random import shuffle
from collections import Counter
import pandas as pd

training_data = np.load('training_data-1.npy')
data_frame = pd.DataFrame(training_data)
print(Counter(data_frame[1].apply(str)))

left_turns = []
right_turns = []
no_turns = []
shuffle(training_data)

for data in training_data:
    image = data[0]
    keystroke = data[1]

    if keystroke == [1, 0, 0]:
        left_turns.append([image, keystroke])
    elif keystroke == [0, 0, 1]:
        right_turns.append([image, keystroke])
    elif keystroke == [0, 1, 0]:
        no_turns.append([image, keystroke])
    else:
        print("no matching keystrokes found")


#no_turns = no_turns[:len(left_turns)][:len(right_turns)]
#left_turns = left_turns[:len(no_turns)]
#right_turns = right_turns[:len(no_turns)]

print(len(no_turns))
print(len(left_turns))
print(len(right_turns))

final_data = no_turns+left_turns+right_turns

shuffle(final_data)
np.save('training_data_final.npy', final_data)
print(len(final_data))





# testing wheather data is correctly loaded or not
# training_data2 = np.load('training_data.npy')
# for data in training_data2:
#     image = data[0]
#     keystroke = data[1]
#     cv2.imshow('trained', image)
#     print(keystroke)
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
