import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

'''
After opening the desired csv file, a formula is grated to 
specifically select each network. This is later used in 
all_data_list to plot a box plot of all points using 
matplotlib. The second plot instead takes all of the test 
data and plots it for each of the test sets. And the final 
plot is the sum of each networks losses for a given test set.
'''

# dis_saves/test_1/testing_2.csv
# file name A,file name B,esn type,number datasets,feedback,test/train,loss

df = pd.read_csv('dis_saves/Final_test/test_1.csv')
# df2 = pd.read_csv('dis_saves/test_1/testing_narma10.csv')

pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)

# mega
filter_mega_test = (df['esn type'] == 'mega_ABA') & (df['test/train'] == 'test')

filter_mega_test_data = pd.DataFrame(df.loc[filter_mega_test, 'loss']).reset_index()
filter_mega_test_data = filter_mega_test_data.drop(['index'], axis=1)

filter_mega_test_data_pip = filter_mega_test_data[filter_mega_test_data.index % 200 < 50]
filter_mega_test_data_rand = filter_mega_test_data[
    (filter_mega_test_data.index % 200 >= 50) & (filter_mega_test_data.index % 200 < 100)]
filter_mega_test_data_narm = filter_mega_test_data[
    (filter_mega_test_data.index % 200 >= 100) & (filter_mega_test_data.index % 200 < 150)]
filter_mega_test_data_simp = filter_mega_test_data[
    (filter_mega_test_data.index % 200 >= 150) & (filter_mega_test_data.index % 200 < 200)]

filter_mega_test_data_pip_list = np.hstack(filter_mega_test_data_pip.values)
filter_mega_test_data_rand_list = np.hstack(filter_mega_test_data_rand.values)
filter_mega_test_data_simp_list = np.hstack(filter_mega_test_data_simp.values)
filter_mega_test_data_narm_list = np.hstack(filter_mega_test_data_narm.values)
# filter_mega_test_list = np.hstack(filter_mega_test_data.values)

# standard_30
filter_standard_30_feedback_test = (df['esn type'] == 'standard_30') & (df['feedback'] == 'yes') & (
            df['test/train'] == 'test')
filter_standard_30_no_feedback_test = (df['esn type'] == 'standard_30') & (df['feedback'] == 'no') & (
            df['test/train'] == 'test')

filter_standard_30_no_feedback_test_data = pd.DataFrame(
    df.loc[filter_standard_30_no_feedback_test, 'loss']).reset_index()
# filter_standard_30_no_feedback_test_list = np.hstack(filter_standard_30_no_feedback_test_data.values)
filter_standard_30_no_feedback_test_data = filter_standard_30_no_feedback_test_data.drop(['index'], axis=1)

filter_standard_30_no_feedback_test_data_pip = filter_standard_30_no_feedback_test_data[
    filter_standard_30_no_feedback_test_data.index % 200 < 50]
filter_standard_30_no_feedback_test_data_rand = filter_standard_30_no_feedback_test_data[
    (filter_standard_30_no_feedback_test_data.index % 200 >= 50) & (
                filter_standard_30_no_feedback_test_data.index % 200 < 100)]
filter_standard_30_no_feedback_test_data_narm = filter_standard_30_no_feedback_test_data[
    (filter_standard_30_no_feedback_test_data.index % 200 >= 100) & (
                filter_standard_30_no_feedback_test_data.index % 200 < 150)]
filter_standard_30_no_feedback_test_data_simp = filter_standard_30_no_feedback_test_data[
    (filter_standard_30_no_feedback_test_data.index % 200 >= 150) & (
                filter_standard_30_no_feedback_test_data.index % 200 < 200)]

filter_standard_30_no_feedback_test_data_pip_list = np.hstack(filter_standard_30_no_feedback_test_data_pip.values)
filter_standard_30_no_feedback_test_data_rand_list = np.hstack(filter_standard_30_no_feedback_test_data_rand.values)
filter_standard_30_no_feedback_test_data_narm_list = np.hstack(filter_standard_30_no_feedback_test_data_narm.values)
filter_standard_30_no_feedback_test_data_simp_list = np.hstack(filter_standard_30_no_feedback_test_data_simp.values)

filter_standard_30_feedback_test_data = pd.DataFrame(df.loc[filter_standard_30_feedback_test, 'loss']).reset_index()
# filter_standard_30_feedback_test_list = np.hstack(filter_standard_30_feedback_test_data.values)
filter_standard_30_feedback_test_data = filter_standard_30_feedback_test_data.drop(['index'], axis=1)

filter_standard_30_feedback_test_data_pip = filter_standard_30_feedback_test_data[
    filter_standard_30_feedback_test_data.index % 200 < 50]
filter_standard_30_feedback_test_data_rand = filter_standard_30_feedback_test_data[
    (filter_standard_30_feedback_test_data.index % 200 >= 50) & (
                filter_standard_30_feedback_test_data.index % 200 < 100)]
filter_standard_30_feedback_test_data_narm = filter_standard_30_feedback_test_data[
    (filter_standard_30_feedback_test_data.index % 200 >= 100) & (
                filter_standard_30_feedback_test_data.index % 200 < 150)]
filter_standard_30_feedback_test_data_simp = filter_standard_30_feedback_test_data[
    (filter_standard_30_feedback_test_data.index % 200 >= 150) & (
                filter_standard_30_feedback_test_data.index % 200 < 200)]

filter_standard_30_feedback_test_data_pip_list = np.hstack(filter_standard_30_feedback_test_data_pip.values)
filter_standard_30_feedback_test_data_rand_list = np.hstack(filter_standard_30_feedback_test_data_rand.values)
filter_standard_30_feedback_test_data_narm_list = np.hstack(filter_standard_30_feedback_test_data_narm.values)
filter_standard_30_feedback_test_data_simp_list = np.hstack(filter_standard_30_feedback_test_data_simp.values)

# standard_60
filter_standard_60_feedback_test = (df['esn type'] == 'standard_60') & (df['feedback'] == 'yes') & (
            df['test/train'] == 'test')
filter_standard_60_no_feedback_test = (df['esn type'] == 'standard_60') & (df['feedback'] == 'no') & (
            df['test/train'] == 'test')

filter_standard_60_no_feedback_test_data = pd.DataFrame(
    df.loc[filter_standard_60_no_feedback_test, 'loss']).reset_index()
# filter_standard_60_no_feedback_test_list = np.hstack(filter_standard_60_no_feedback_test_data.values)
filter_standard_60_no_feedback_test_data = filter_standard_60_no_feedback_test_data.drop(['index'], axis=1)

filter_standard_60_no_feedback_test_data_pip = filter_standard_60_no_feedback_test_data[
    filter_standard_60_no_feedback_test_data.index % 200 < 50]
filter_standard_60_no_feedback_test_data_rand = filter_standard_60_no_feedback_test_data[
    (filter_standard_60_no_feedback_test_data.index % 200 >= 50) & (
                filter_standard_60_no_feedback_test_data.index % 200 < 100)]
filter_standard_60_no_feedback_test_data_narm = filter_standard_60_no_feedback_test_data[
    (filter_standard_60_no_feedback_test_data.index % 200 >= 100) & (
                filter_standard_60_no_feedback_test_data.index % 200 < 150)]
filter_standard_60_no_feedback_test_data_simp = filter_standard_60_no_feedback_test_data[
    (filter_standard_60_no_feedback_test_data.index % 200 >= 150) & (
                filter_standard_60_no_feedback_test_data.index % 200 < 200)]

filter_standard_60_no_feedback_test_data_pip_list = np.hstack(filter_standard_60_no_feedback_test_data_pip.values)
filter_standard_60_no_feedback_test_data_rand_list = np.hstack(filter_standard_60_no_feedback_test_data_rand.values)
filter_standard_60_no_feedback_test_data_narm_list = np.hstack(filter_standard_60_no_feedback_test_data_narm.values)
filter_standard_60_no_feedback_test_data_simp_list = np.hstack(filter_standard_60_no_feedback_test_data_simp.values)

filter_standard_60_feedback_test_data = pd.DataFrame(df.loc[filter_standard_60_feedback_test, 'loss']).reset_index()
# filter_standard_60_feedback_test_list = np.hstack(filter_standard_60_feedback_test_data.values)
filter_standard_60_feedback_test_data = filter_standard_60_feedback_test_data.drop(['index'], axis=1)

filter_standard_60_feedback_test_data_pip = filter_standard_60_feedback_test_data[
    filter_standard_60_feedback_test_data.index % 200 < 50]
filter_standard_60_feedback_test_data_rand = filter_standard_60_feedback_test_data[
    (filter_standard_60_feedback_test_data.index % 200 >= 50) & (
                filter_standard_60_feedback_test_data.index % 200 < 100)]
filter_standard_60_feedback_test_data_narm = filter_standard_60_feedback_test_data[
    (filter_standard_60_feedback_test_data.index % 200 >= 100) & (
                filter_standard_60_feedback_test_data.index % 200 < 150)]
filter_standard_60_feedback_test_data_simp = filter_standard_60_feedback_test_data[
    (filter_standard_60_feedback_test_data.index % 200 >= 150) & (
                filter_standard_60_feedback_test_data.index % 200 < 200)]

filter_standard_60_feedback_test_data_pip_list = np.hstack(filter_standard_60_feedback_test_data_pip.values)
filter_standard_60_feedback_test_data_rand_list = np.hstack(filter_standard_60_feedback_test_data_rand.values)
filter_standard_60_feedback_test_data_narm_list = np.hstack(filter_standard_60_feedback_test_data_narm.values)
filter_standard_60_feedback_test_data_simp_list = np.hstack(filter_standard_60_feedback_test_data_simp.values)

# double_30
filter_double_30_feedback_test = (df['esn type'] == 'double_30') & (df['feedback'] == 'yes') & (
            df['test/train'] == 'test')
filter_double_30_no_feedback_test = (df['esn type'] == 'double_30') & (df['feedback'] == 'no') & (
            df['test/train'] == 'test')

filter_double_30_no_feedback_test_data = pd.DataFrame(df.loc[filter_double_30_no_feedback_test, 'loss']).reset_index()
# filter_double_30_no_feedback_test_list = np.hstack(filter_double_30_no_feedback_test_data.values)
filter_double_30_no_feedback_test_data = filter_double_30_no_feedback_test_data.drop(['index'], axis=1)

filter_double_30_no_feedback_test_data_pip = filter_double_30_no_feedback_test_data[
    filter_double_30_no_feedback_test_data.index % 200 < 50]
filter_double_30_no_feedback_test_data_rand = filter_double_30_no_feedback_test_data[
    (filter_double_30_no_feedback_test_data.index % 200 >= 50) & (
                filter_double_30_no_feedback_test_data.index % 200 < 100)]
filter_double_30_no_feedback_test_data_narm = filter_double_30_no_feedback_test_data[
    (filter_double_30_no_feedback_test_data.index % 200 >= 100) & (
                filter_double_30_no_feedback_test_data.index % 200 < 150)]
filter_double_30_no_feedback_test_data_simp = filter_double_30_no_feedback_test_data[
    (filter_double_30_no_feedback_test_data.index % 200 >= 150) & (
                filter_double_30_no_feedback_test_data.index % 200 < 200)]

filter_double_30_no_feedback_test_data_pip_list = np.hstack(filter_double_30_no_feedback_test_data_pip.values)
filter_double_30_no_feedback_test_data_rand_list = np.hstack(filter_double_30_no_feedback_test_data_rand.values)
filter_double_30_no_feedback_test_data_narm_list = np.hstack(filter_double_30_no_feedback_test_data_narm.values)
filter_double_30_no_feedback_test_data_simp_list = np.hstack(filter_double_30_no_feedback_test_data_simp.values)

filter_double_30_feedback_test_data = pd.DataFrame(df.loc[filter_double_30_feedback_test, 'loss']).reset_index()
# filter_double_30_feedback_test_list = np.hstack(filter_double_30_feedback_test_data.values)
filter_double_30_feedback_test_data = filter_double_30_feedback_test_data.drop(['index'], axis=1)

filter_double_30_feedback_test_data_pip = filter_double_30_feedback_test_data[
    filter_double_30_feedback_test_data.index % 200 < 50]
filter_double_30_feedback_test_data_rand = filter_double_30_feedback_test_data[
    (filter_double_30_feedback_test_data.index % 200 >= 50) & (filter_double_30_feedback_test_data.index % 200 < 100)]
filter_double_30_feedback_test_data_narm = filter_double_30_feedback_test_data[
    (filter_double_30_feedback_test_data.index % 200 >= 100) & (filter_double_30_feedback_test_data.index % 200 < 150)]
filter_double_30_feedback_test_data_simp = filter_double_30_feedback_test_data[
    (filter_double_30_feedback_test_data.index % 200 >= 150) & (filter_double_30_feedback_test_data.index % 200 < 200)]

filter_double_30_feedback_test_data_pip_list = np.hstack(filter_double_30_feedback_test_data_pip.values)
filter_double_30_feedback_test_data_rand_list = np.hstack(filter_double_30_feedback_test_data_rand.values)
filter_double_30_feedback_test_data_narm_list = np.hstack(filter_double_30_feedback_test_data_narm.values)
filter_double_30_feedback_test_data_simp_list = np.hstack(filter_double_30_feedback_test_data_simp.values)

all_data_list = np.dstack((filter_mega_test_data_pip_list, filter_mega_test_data_rand_list,
                           filter_mega_test_data_narm_list, filter_mega_test_data_simp_list,
                           filter_standard_30_no_feedback_test_data_pip_list,
                           filter_standard_30_no_feedback_test_data_rand_list,
                           filter_standard_30_no_feedback_test_data_narm_list,
                           filter_standard_30_no_feedback_test_data_simp_list,
                           filter_standard_30_feedback_test_data_pip_list,
                           filter_standard_30_feedback_test_data_rand_list,
                           filter_standard_30_feedback_test_data_narm_list,
                           filter_standard_30_feedback_test_data_simp_list,
                           filter_standard_60_no_feedback_test_data_pip_list,
                           filter_standard_60_no_feedback_test_data_rand_list,
                           filter_standard_60_no_feedback_test_data_narm_list,
                           filter_standard_60_no_feedback_test_data_simp_list,
                           filter_standard_60_feedback_test_data_pip_list,
                           filter_standard_60_feedback_test_data_rand_list,
                           filter_standard_60_feedback_test_data_narm_list,
                           filter_standard_60_feedback_test_data_simp_list,
                           filter_double_30_no_feedback_test_data_pip_list,
                           filter_double_30_no_feedback_test_data_rand_list,
                           filter_double_30_no_feedback_test_data_narm_list,
                           filter_double_30_no_feedback_test_data_simp_list,
                           filter_double_30_feedback_test_data_pip_list, filter_double_30_feedback_test_data_rand_list,
                           filter_double_30_feedback_test_data_narm_list, filter_double_30_feedback_test_data_simp_list
                           ))

pip_data_list = np.dstack((filter_mega_test_data_pip_list, filter_standard_30_no_feedback_test_data_pip_list,
                           filter_standard_30_feedback_test_data_pip_list,
                           filter_standard_60_no_feedback_test_data_pip_list,
                           filter_standard_60_feedback_test_data_pip_list,
                           filter_double_30_no_feedback_test_data_pip_list,
                           filter_double_30_feedback_test_data_pip_list,))
rand_data_list = np.dstack((filter_mega_test_data_rand_list, filter_standard_30_no_feedback_test_data_rand_list,
                            filter_standard_30_feedback_test_data_rand_list,
                            filter_standard_60_no_feedback_test_data_rand_list,
                            filter_standard_60_feedback_test_data_rand_list,
                            filter_double_30_no_feedback_test_data_rand_list,
                            filter_double_30_feedback_test_data_rand_list,))
narm_data_list = np.dstack((filter_mega_test_data_narm_list, filter_standard_30_no_feedback_test_data_narm_list,
                            filter_standard_30_feedback_test_data_narm_list,
                            filter_standard_60_no_feedback_test_data_narm_list,
                            filter_standard_60_feedback_test_data_narm_list,
                            filter_double_30_no_feedback_test_data_narm_list,
                            filter_double_30_feedback_test_data_narm_list,))
simp_data_list = np.dstack((filter_mega_test_data_simp_list, filter_standard_30_no_feedback_test_data_simp_list,
                            filter_standard_30_feedback_test_data_simp_list,
                            filter_standard_60_no_feedback_test_data_simp_list,
                            filter_standard_60_feedback_test_data_simp_list,
                            filter_double_30_no_feedback_test_data_simp_list,
                            filter_double_30_feedback_test_data_simp_list,))

# Example of the Student's t-test
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel

data1 = filter_mega_test_data_pip_list

means = []
stds = []

for data2 in np.swapaxes(pip_data_list.reshape(-1, 7),0,1) :
    means.append(np.mean(data2))
    stds.append(np.std(data2))

    stat, p = ttest_ind(data1, data2, equal_var=False)
    print(f'stat={stat}, p={p}')
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

print(f'Means: {means}')
print(f'Standard Deviations: {stds}')

print(np.mean(filter_standard_30_no_feedback_test_data_pip_list))

if True:
    x = 10
    y = 8
    all_data_list = all_data_list.reshape(-1, 28)

    plt.rcParams["figure.figsize"] = (x,y)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Box plot showing each ESN type and the data collected from 10 iterations of 50 tests on 3 datasets')
    ax1.boxplot(pd.DataFrame(all_data_list), vert=True, notch=True)
    ax1.set_xlabel('Mean Squared Error')
    ax1.set_yticklabels(
        ['Auto Chain ESN - Complex Data', 'Auto Chain ESN - Random Data', 'Auto Chain ESN - NARMA10 Data',
         'Auto Chain ESN - Simple Data',
         'standard 30 no feedback', 'standard 30 no feedback', 'standard 30 no feedback', 'standard 30 no feedback',
         'standard 30 feedback', 'standard 30 feedback', 'standard 30 feedback', 'standard 30 feedback',
         'standard 60 no feedback', 'standard 60 no feedback', 'standard 60 no feedback', 'standard 60 no feedback',
         'standard 60 feedback', 'standard 60 feedback', 'standard 60 feedback', 'standard 60 feedback',
         'double 30 no feedback', 'double 30 no feedback', 'double 30 no feedback', 'double 30 no feedback',
         'double 30 feedback', 'double 30 feedback', 'double 30 feedback', 'double 30 feedback'])
    ##plt.show()

    pip_data_list = pip_data_list.reshape(-1, 7)

    plt.rcParams["figure.figsize"] =(x,y)
    fig2, ax1 = plt.subplots()
    ax1.set_title('Box plot showing each ESN type and the data\ncollected from 50 test sets\nfrom 50 test networks, '
                  'using the pattern within a pattern dataset', fontsize=18)
    ax1.boxplot(pd.DataFrame(pip_data_list), vert=False, notch=True, showfliers=False)
    ax1.set_xlabel('Mean Squared Error', fontsize=18)
    ax1.set_yticklabels(['AC', 'S30', 'S30-F', 'S60', 'S60-F', '2C-S30', '2C-S30-F'], fontsize=18)
    plt.show()

    rand_data_list = rand_data_list.reshape(-1, 7)

    plt.rcParams["figure.figsize"] = (x,y)
    fig3, ax1 = plt.subplots()
    ax1.set_title('Box plot showing each ESN type and the data\ncollected from 50 test sets\nfrom 50 test networks, '
                  'using the random value dataset', fontsize=18)
    ax1.boxplot(pd.DataFrame(rand_data_list), vert=False, notch=True, showfliers=False)
    ax1.set_xlabel('Mean Squared Error', fontsize=18)
    ax1.set_yticklabels(['AC', 'S30', 'S30-F', 'S60', 'S60-F', '2C-S30', 'n2C-S30-F'], fontsize=18)
    plt.show()

    narm_data_list = narm_data_list.reshape(-1, 7)

    plt.rcParams["figure.figsize"] = (x,y)
    fig4, ax1 = plt.subplots()
    ax1.set_title('Box plot showing each ESN type and the data\ncollected from 50 test sets\nfrom 50 test networks, '
                  'using the NARMA10 dataset', fontsize=18)
    ax1.boxplot(pd.DataFrame(narm_data_list), vert=False, notch=True, showfliers=False)
    ax1.set_xlabel('Mean Squared Error', fontsize=18)
    ax1.set_yticklabels(['AC', 'S30', 'S30-F', 'S60', 'S60-F', '2C-S30', '2C-S30-F'], fontsize=18)
    plt.show()

    simp_data_list = simp_data_list.reshape(-1, 7)

    plt.rcParams["figure.figsize"] =(x,y)
    fig4, ax1 = plt.subplots()
    ax1.set_title('Box plot showing each ESN type and the data collected from 50 test sets from 50 test networks, '
                  'using the simple dataset')
    simple_data_list_formatted = [a / 10000 for a in simp_data_list]
    ax1.boxplot(pd.DataFrame(simple_data_list_formatted), vert=True, notch=True, showfliers=False)
    ax1.set_xlabel('Mean Squared Error')
    ax1.set_yticklabels(['AC', 'S30', 'S30-F', 'S60', 'S60-F', '2C-S30' , '2C-S30-F'])
    ##plt.show()

    filter_mega_test_data_pip_list = filter_mega_test_data_pip_list.reshape(-1, 50)

    plt.rcParams["figure.figsize"] =(x,y)
    fig5, ax1 = plt.subplots()
    ax1.set_title('Box plot showing each test sets results for the loop back esn')
    ax1.boxplot(pd.DataFrame(filter_mega_test_data_pip_list), vert=True, notch=True)
    ax1.set_xlabel('Mean Squared Error')
    ax1.set_yticklabels([x + 1 for x in range(50)])
    ##plt.show()
