from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
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


def show_difference(kmeans, word):

    # gpd mean pib in French
    sns.scatterplot(kmeans[word], kmeans['gdpp'], hue='KMeans_Clusters', data=kmeans)

    # make the first character of the word to uppercase and replace any _ with a space
    title = word.title().replace('_', ' ')
    title = title[0].title() + title[1:]

    plt.title(title + " vs gdpp", fontsize=15)
    plt.xlabel(word, fontsize=12)
    plt.ylabel("gdpp", fontsize=12)
    plt.show()


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
    print(scaled.head())

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

    show_difference(kmeans_data, 'child_mort')

    return 0


if __name__ == '__main__':
    main()
