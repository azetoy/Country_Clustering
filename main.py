import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, Birch
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def correlation(data):
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.show()
    # as we can see from the heatmap, there are some highly correlated features gdpp , income, imports and exports,
    # child_mortality and total_fert are highly correlated (as they increase together)

    # we can also see in the heat map life_expec and child_mortality are highly negatively correlated (as they decrease
    # together)


def k_means(scaled, df):
    word = 'income'
    # by using elbow method we can find the optimal number of clusters for the data (3)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(scaled)

    # now we are going to see if the model is good by using the silhouette coefficient as the coefficient
    # is 0.28 the model is average good
    # print(metrics.silhouette_score(scaled, kmeans.labels_))

    # predict the clusters
    kmeans.fit_predict(scaled, kmeans.labels_)
    prediction = kmeans.labels_
    kmeans_data = pd.DataFrame(df)
    kmeans_data['KMeans_Clusters'] = prediction
    # we save the dataframe to csv file
    kmeans_data.to_csv('data_csv/kmeans_result.csv', index=False)

    sns.scatterplot(x=kmeans_data[word], y=kmeans_data.gdpp, hue='KMeans_Clusters', data=kmeans_data)

    # make the first character of the word to uppercase and replace any _ with a space
    title = word.title().replace('_', ' ')
    title = title[0].title() + title[1:]

    plt.title(title + " vs gdpp", fontsize=15)
    plt.xlabel(word, fontsize=12)
    plt.ylabel("gdpp", fontsize=12)
    plt.savefig("Plots/{}.png".format("Kmeans_Clusters_" + word))
    plt.show()


def d_bscan(data, df):
    word = 'income'
    # we replace any null value to number and create a numpy array
    # with the data, and finally we scale the data
    data = np.nan_to_num(data)
    data = np.array(data, dtype=np.float64)
    data = StandardScaler().fit_transform(data)

    db = DBSCAN(eps=1, min_samples=3).fit(data)

    datacopy = df.copy()
    datacopy['DB_cluster'] = db.labels_
    datacopy.to_csv('data_csv/DB_SCAN.csv', index=False)

    plt.figure(figsize=(12, 6))
    plt.scatter(datacopy['income'], datacopy['gdpp'], c=db.labels_)
    title = word.title().replace('_', ' ')
    title = title[0].title() + title[1:]
    plt.title(title + ' vs GDPP ', fontsize=15)
    plt.xlabel(word, fontsize=12)
    plt.ylabel("GDPP", fontsize=12)
    plt.savefig("Plots/{}.png".format("DBSCAN_Clusters_" + word))
    plt.show()


def birch(scaled, df):
    birch = Birch(n_clusters=5).fit(scaled)
    pred_birch = birch.fit_predict(scaled)

    clustered_data = df.copy()
    clustered_data["cluster_index"] = pred_birch

    sns.scatterplot(x=clustered_data.gdpp,
                    y=clustered_data.income,
                    hue=clustered_data.cluster_index,
                    palette="deep")
    plt.savefig("data_csv/Birch.png")


def main():
    # Read data csv file
    # we also drop the country name column as we do unsupervised learning we don't need it
    df = pd.read_csv('data_csv/Country-data.csv')
    data = pd.read_csv('data_csv/Country-data.csv').drop(['country'], axis=1)
    # dict_df = pd.read_csv('data_csv/data-dictionary.csv')

    # correlation(data)

    # we need to scale the data because some values are much bigger than others
    # we use StandardScaler to scale the data, but we can also use MinMaxScaler

    scaling = StandardScaler()
    scaled = pd.DataFrame(scaling.fit_transform(data), columns=data.columns)

    # we use the k means function to use the k means algorithm to the data set

    # k_means(scaled, df)

    # we use the dbscan function to use the dbscan algorithm to the data set if you make the dbscan function call
    # after the k means function call it will give you the k means clusters and the dbscan clusters in the
    # dbscan_result.csv file the same thing will happen if you reverse the call this time the two cluster data will
    # be in the kmeans_result.csv file

    k_means(scaled, df)
    # d_bscan(data, df)
    # birch(scaled, df)
    return 0


if __name__ == '__main__':
    main()
