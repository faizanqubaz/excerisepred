from matplotlib import pyplot as plt
import seaborn as sns


def check_distribution(df):
    univariate_analysis(df)
    multivariate_analysis(df)




def univariate_analysis(df):
    for col in df.columns:
        plt.figure(figsize=(12,12))
        plt.subplot(111)
        sns.histplot(df[col])

        plt.subplot(122)
        sns.distplot(df[col])
        plt.show()


def multivariate_analysis(df):
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):

            column1 = df.columns[i]
            column2 = df.columns[j]

            if ((df[column1].dtypes == 'int64' or df[column1].dtypes == 'float64') and (df[column2].dtypes == 'int64' or df[column2].dtypes == 'float64')):
                plt.figure(figsize=(12,12))
                sns.scatterplot(x=df[column1],y=df[column2],data=df)
                plt.show()

