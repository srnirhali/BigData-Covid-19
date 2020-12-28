from datetime import datetime, timedelta
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark import SparkContext
import pandas as pd
from pyspark.sql.functions import *
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import geopandas
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

# ######### Big Data Assignment ######### #
# ######### Shivam Rajdendra Nirhali : 219203376 ######### #
# ######### Abhishek Dave : 219203366 ######### #
# ######### Dhaval Mansukhbhai Karkar : 219203375 ######### #
# ######### Gayatri Tawada : 219203030  ######### #

# https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases

# getFirstCaseDate will return the date on which first case occurred in country data-set
def getFirstCaseDate(spark_country):
    return spark_country.filter("cases != 0").first()['date']


# getLastDate will return the last date in data-set
def getLastDate(spark_country):
    return spark_country.withColumn("date", F.col("date")).groupBy("iso_a3").agg(expr("max(date) as max")).first()[1]


# getSparkDataFrame will return the Spark DataFrame with date column from pandas DataFrame.
def getSparkDataFrame(pandas_df):
    mySchema = StructType([StructField("dateRep", DateType(), True) \
                              , StructField("day", IntegerType(), True) \
                              , StructField("month", IntegerType(), True) \
                              , StructField("year", IntegerType(), True) \
                              , StructField("cases", IntegerType(), True) \
                              , StructField("deaths", IntegerType(), True) \
                              , StructField("countriesAndTerritories", StringType(), True) \
                              , StructField("geoId", StringType(), True) \
                              , StructField("iso_a3", StringType(), True) \
                              , StructField("popData2019", FloatType(), True) \
                              , StructField("continentExp", StringType(), True)])

    spark_df = sql_sc.createDataFrame(pandas_df, schema=mySchema)
    spark_df = spark_df.withColumn("date", to_date(F.concat_ws("-", "year", "month", "day")))
    return spark_df


# getSplitDFWithDateIndexBetweenDates will return spark Dataframe with date index
def getSplitDFWithDateIndexBetweenDates(spark_country, dates):
    temp = spark_country.filter(F.col('date').between(*dates)).withColumn("dateIndex",
                                                                datediff(to_date("date", "yyyy-MM-dd"),
                                                                to_date(F.lit(dates[0])))).sort('date',ascending=True)
    new_date = dates[1] - timedelta(1)
    s_train_df = temp.filter((F.col('date') != dates[1]) & (F.col('date') != new_date)).sort(
        'dateIndex', ascending=True)
    s_test_df = temp.filter((F.col('date') == dates[1]) | (F.col('date') == new_date))
    return s_train_df, s_test_df


# plotCasesAndDeathsOnGeo will plot the total cases and deaths by country on geo plot.
def plotCasesAndDeathsOnGeo(spark_df):
    total = s_df.groupBy("iso_a3").agg(expr("sum(cases) as cases"), expr("sum(deaths) as deaths"))
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    country_shapes = world[['geometry', 'iso_a3']]
    country_shapes = country_shapes.merge(total.toPandas(), on='iso_a3')
    country_shapes.plot(column='cases', legend=True)
    country_shapes.plot(column='deaths', legend=True)


# applyRegression will predict and plot the cases of test data-set on the bases of train data
def applyRegression(train, test):
    vectorAssembler = VectorAssembler(inputCols=['dateIndex'], outputCol='features')
    data = vectorAssembler.transform(train)
    train = data.select(['features', 'cases', 'dateIndex', 'date'])
    data = vectorAssembler.transform(test)
    test = data.select(['features', 'cases', 'dateIndex', 'date'])
    lr = LinearRegression(featuresCol='features', labelCol='cases')
    lr_model = lr.fit(train)
    lr_p = lr_model.transform(test)
    lr_p.select("prediction", "cases", "features", 'date').show()
    x1 = train.toPandas()['dateIndex'].astype(int).values.tolist()
    y1 = train.toPandas()['cases'].values.tolist()
    x2 = lr_p.toPandas()['dateIndex'].astype(int).values.tolist()
    y2 = lr_p.toPandas()['prediction'].values.tolist()
    y_2 = lr_p.toPandas()['cases'].values.tolist()
    x1.extend(x2)

    y1.extend(y_2)
    plt.scatter(x1, y1, color='blue', s=30)
    abline(lr_model.coefficients[0], lr_model.intercept)
    plt.scatter(x2, y2, color='red', s=30)


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def datesToString(dates):
    return  dates[0].strftime("%x")+" to " +dates[1].strftime("%x")

# will plot the data of country with three different plots
# last 2 days will be used for prediction.
# 1.First case till last date,
# 2. prev (train + test) 60 + 2 days from last date,
# 3.prev (train + test) 14 + 2 days from last date
def plotData(data,iso_a3):
    country = data.filter("iso_a3=='"+iso_a3+"'").sort('date', ascending=True)

    first_date = getFirstCaseDate(country)
    max_date = getLastDate(country)
    dates = (first_date, max_date)
    plt.figure(datesToString(dates))
    plt.xlabel("Date Index")
    plt.ylabel("Cases")
    plt.title(iso_a3 + " : Regression on First case till last Date  ")
    s_train_df, s_test_df = getSplitDFWithDateIndexBetweenDates(country, dates)
    applyRegression(s_train_df, s_test_df)

    min_date = max_date - timedelta(62)
    dates = (min_date, max_date)
    plt.figure(datesToString(dates))
    plt.xlabel("Date Index")
    plt.ylabel("Cases")
    plt.title(iso_a3 + " : Regression on prev (train + test) 60 + 2 days from last date")
    s_train_df, s_test_df = getSplitDFWithDateIndexBetweenDates(country, dates)
    applyRegression(s_train_df, s_test_df)

    min_date = max_date - timedelta(16)
    dates = (min_date, max_date)
    plt.figure(datesToString(dates))
    plt.xlabel("Date Index")
    plt.ylabel("Cases")
    plt.title(iso_a3 + " : Regression on prev (train + test) 14 + 2 days from last date")
    s_train_df, s_test_df = getSplitDFWithDateIndexBetweenDates(country, dates)
    applyRegression(s_train_df, s_test_df)

    dates = (datetime(2020,2,1), datetime(2020,3,23))
    plt.figure(datesToString(dates))
    plt.title(iso_a3 + " : Regression on : " +datesToString(dates))
    plt.xlabel("Date Index")
    plt.ylabel("Cases")
    s_train_df, s_test_df = getSplitDFWithDateIndexBetweenDates(country, dates)
    applyRegression(s_train_df, s_test_df)
    plt.show()


if __name__ == '__main__':
    sc = SparkContext('local', 'example')  # if using locally
    sc.setLogLevel("Error")
    sql_sc = SQLContext(sc)
    # https://www.ecdc.europa.eu/en/geographical-distribution-2019-ncov-cases
    pandas_df = pd.read_csv('Data\covid19.csv')
    pandas_df['dateRep'] = pandas_df['dateRep'].astype('datetime64[ns]')
    # convert pandas dataFrame to spark dataframe
    s_df = getSparkDataFrame(pandas_df);
    # display First five rows
    s_df.show(5)
    # will plot Geo Map of total cases and total deaths
    plotCasesAndDeathsOnGeo(s_df)

    # will plot the data of country with three different plots
    # 1.First case till last date,
    # 2. prev 60 days from last date,
    # 3.prev 14 days from last date
    plotData(s_df,'AUS')
    # plotData(s_df, 'DEU')