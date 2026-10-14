from sklearn.linear_model import LinearRegression
import random

bream_length = [
    24.8, 25.7, 26.1, 27.2, 27.8,
    28.4, 28.9, 29.3, 29.8, 30.3,
    30.8, 31.3, 31.7, 32.2, 32.6,
    33.1, 33.6, 34.1, 34.6, 35.2,
    35.6, 36.2, 36.7, 37.3, 37.8,
    38.2, 38.8, 39.2, 39.8, 40.2,
    40.7, 41.3, 41.8, 42.4, 42.9
]
bream_weight = [
    225.0, 265.0, 315.0, 350.0, 385.0,
    410.0, 445.0, 475.0, 510.0, 535.0,
    555.0, 585.0, 605.0, 625.0, 655.0,
    680.0, 705.0, 735.0, 760.0, 795.0,
    815.0, 840.0, 865.0, 890.0, 925.0,
    950.0, 980.0, 1015.0, 1040.0, 1075.0,
    1100.0, 1135.0, 1160.0, 1200.0, 1235.0
]
smelt_length = [
    9.6, 10.1, 10.4, 10.8, 11.1,
    11.5, 11.7, 12.1, 12.3, 12.6,
    12.9, 13.2, 13.8, 14.6
]
smelt_weight = [
    5.9, 6.8, 7.4, 8.2, 8.9,
    9.3, 9.7, 10.6, 11.3, 12.0,
    13.1, 14.0, 16.8, 18.5
]
length = bream_length + smelt_length
weight = bream_weight + smelt_weight
data = [[l,w] for l, w in zip(length, weight)]
data = random.shuffle(data)
train_input = [d[0] for d in data[:35]]
train_target = [d[1] for d in data[:35]]
test_input = [d[0] for d in data[35:]]
test_target = [d[1] for d in data[35:]]

lr = LinearRegression()
lr.fit(train_input, train_target)
