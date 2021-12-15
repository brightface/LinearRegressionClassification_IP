import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('src_ip_to_dst.txt')
sys.stdin = open("src_ip_to_dst.txt")

first_attribute = []
answer = []

for i in range(len(df)):
    a,b,c = list(map(int,input().split()))
    first_attribute.append(a)
    answer.append(c)
attribute = np.array(first_attribute)
ans = np.array(answer)

train_input, test_input, train_target, test_target = train_test_split(
    attribute, ans, random_state=42
)

train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

train_poly = np.column_stack((train_input**2 , train_input))
test_poly = np.column_stack((test_input**2, test_input))

# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# ss.fit(train_input)
# train_scaled = ss.transform(train_input)
# test_scaled = ss.transform(test_input)
# kn = KNeighborsClassifier(n_neighbors=3)
# kn.fit(train_scaled,train_target)
# print(kn.score(train_scaled,train_target))
# print(kn.score(test_scaled,test_target))


lr = LogisticRegression()
lr.fit(train_input,train_target) # 선형회귀 55 56
print(lr.score(train_input,train_target))
print(lr.score(test_input,test_target))
print(lr.coef_, lr.intercept_)
print()
lr.fit(train_poly, train_target) #다항회귀 55 56

# print(lr.classes_)
#선형휘귀는 test 55 train56
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))

print(lr.coef_, lr.intercept_)
plt.scatter(train_input, train_target)
# plt.scatter(train_poly,train_target)
# plt.plot()
plt.xlabel('Sorted IP')
plt.ylabel('strange seek signal')
plt.show()