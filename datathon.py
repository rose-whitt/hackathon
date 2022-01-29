from pandas.core.frame import DataFrame
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import collections
import numpy
# import seaborn as sns

data = pd.read_csv("track.csv")

print(data.head(8))

print(data.dtypes)

# convert categories to numerical values
# data['Income'].unique()
print("range is", data.income.max()-data.income.min())
print("max:", data.income.max())
print("min:", data.income.min())
print('mean:', data.income.mean())


new_data = data.filter(['applicant_age', 'race', 'ethnicity', 'sex',
                        'income', 'debt_to_income_ratio', 'loan_amount', 'accepted'])
# eliminate w/missing value
new_data.isna().sum()
new_data.dropna(inplace=True)
print(new_data.head(5))


new_data['loan_to_income_ratio'] = (
    new_data['loan_amount']/1000)/new_data['income']
print(new_data.head(10))
print(new_data['loan_to_income_ratio'].head(20))
print(new_data['loan_to_income_ratio'].median())
print(new_data['loan_to_income_ratio'].mean())
print(new_data['loan_to_income_ratio'].max())
print(new_data['loan_to_income_ratio'].min())

print("----------------------")
print(type(new_data['race']))
print(new_data['race'].size)
print(new_data['race'].unique())

y = np.array([35, 25, 25, 15])

race_np_arr = new_data['race'].values
print(type(race_np_arr))
print(collections.Counter(race_np_arr))
print()
white_count = np.count_nonzero(race_np_arr == "White")
black_count = np.count_nonzero(race_np_arr == "Black or African American")
asian_count = np.count_nonzero(race_np_arr == "Asian")
joint_count = np.count_nonzero(race_np_arr == "Joint")
NA_count = np.count_nonzero(race_np_arr == "American Indian or Alaska Native")
two_count = np.count_nonzero(race_np_arr == "2 or more minority races")
island_count = np.count_nonzero(
    race_np_arr == "Native Hawaiian or Other Pacific Islander")
free_count = np.count_nonzero(race_np_arr == "Free Form Text Only")

arr = np.array([white_count, black_count, asian_count,
                joint_count, NA_count, two_count, island_count, free_count])
plt.pie(arr)
# plt.show()

print("White Count", white_count)
print("Black Count: ", black_count)
print("!!!!!!!!!!!!!")
print()
count_white = race_np_arr.tolist().count('White"')
count_black = race_np_arr.tolist().count("Black or African American")
print("count white: ", count_white)

# plt.pie(y)
# plt.show()

print(type(new_data['accepted']))
print(new_data['accepted'].size)

accepted_arr = new_data['accepted'].values

# given race string
# return array of [accepted count, deny count]


def count_race_accept_race(race, accept_arr, race_arr):
    race_acceptance = []
    accept_count = 0
    deny_count = 0
    idx = 0
    # iterate over races
    for elem in race_arr:
        # if a race match
        if (elem == race):
            # determine accept value
            if accept_arr[idx] == 1.0:
                accept_count += 1
            else:
                deny_count += 1
        idx += 1
    return [accept_count, deny_count]


accept_labels = ['Accepted', 'Denied']
colors = ['#2a3990', '#d23369']

white_pie = count_race_accept_race("White", accepted_arr, race_np_arr)
white_denial_percent = (white_pie[1] / (white_pie[0] + white_pie[1]))*100
print("WHITE DENIAL PERCENT: ", white_denial_percent)
plt.pie(white_pie, labels=accept_labels, autopct='%.1f%%', colors=colors)
plt.title("Acceptance by Race: White")
plt.savefig('white_pie.png')
plt.show()

black_pie = count_race_accept_race(
    "Black or African American", accepted_arr, race_np_arr)
black_denial_percent = (black_pie[1] / (black_pie[0] + black_pie[1]))*100
print("BLACK DENIAL PERCENT: ", black_denial_percent)
plt.pie(black_pie, labels=accept_labels, autopct='%.1f%%', colors=colors)
plt.title("Acceptance by Race: Black or African American")
plt.savefig('black_pie.png')
plt.show()

asian_pie = count_race_accept_race("Asian", accepted_arr, race_np_arr)
asian_denial_percent = (asian_pie[1] / (asian_pie[0] + asian_pie[1]))*100
print("ASIAN DENIAL PERCENT: ", asian_denial_percent)
plt.pie(asian_pie, labels=accept_labels, autopct='%.1f%%', colors=colors)
plt.title("Acceptance by Race: Asian")
plt.savefig('asian_pie.png')
plt.show()

na_pie = count_race_accept_race(
    "American Indian or Alaska Native", accepted_arr, race_np_arr)
na_denial_percent = (na_pie[1] / (na_pie[0] + na_pie[1]))*100
print("NA DENIAL PERCENT: ", na_denial_percent)
plt.pie(na_pie, labels=accept_labels, autopct='%.1f%%', colors=colors)
plt.title("Acceptance by Race: American Indian or Alaska Native")
plt.savefig('na_pie.png')
plt.show()

two_pie = count_race_accept_race(
    "2 or more minority races", accepted_arr, race_np_arr)
two_denial_percent = (two_pie[1] / (two_pie[0] + two_pie[1]))*100
print("NA DENIAL PERCENT: ", two_denial_percent)
plt.pie(two_pie, labels=accept_labels, autopct='%.1f%%', colors=colors)
plt.title("Acceptance by Race: 2 or more minority races")
plt.savefig('two_pie.png')
plt.show()


# BAR CHART
# create dataset
bar_height = [white_denial_percent, black_denial_percent,
              asian_denial_percent, na_denial_percent, two_denial_percent]
bars = ['W', 'B', 'A',
        'NA', '2']
x_pos = np.arange(len(bars))

# create bars
plt.bar(x_pos, bar_height, color=(0.2, 0.4, 0.6, 0.6))

# create names on the x-axis
plt.xticks(x_pos, bar_height)

# show graph
plt.savefig('race_bar.png')
plt.show()
print("-----------")

####
fig, ax = plt.subplots()
# ax.stackplot(new_data['loan_to_income_ratio'], logistic regression data, )


def prob_dist(inc):
    mean = np.mean(inc)
    stand_dev = np.std(inc)
    y_func = 1/(stand_dev*np.sqrt(2 * np.pi)) * np.exp(-(inc-mean))
    return y_func


# income_numpy = data['income'].to_numpy()
# print(income_numpy)


# x_val = np.arange(-2, 2, 0.1)
data2 = np.array(data[1:])
# print(data2)


# def ratio(arr):
# for each income
# get the loan amount at that index
#
# y_val = prob_dist(x_val)

# plot bell curve
# plt.style.use('seaborn')
#plt.figure(figsize=(6, 6))
#plt.plot(x_val, y_val, color='red')
# plt.show()

# income_sort =d
