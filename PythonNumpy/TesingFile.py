from datetime import datetime,timedelta
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SQLContext
import pandas as pd
from pyspark.sql.functions import *
import geopandas
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import geopandas

import os

from pyspark.ml.regression import LinearRegression

from pyspark.ml.feature import VectorAssembler

sc = SparkContext('local','example')  # if using locally
sc.setLogLevel("Error")
sql_sc = SQLContext(sc)
test = sc.textFile('Data\covid19.csv')
#filtered = test.filter(test("date").lt(lit("2015-03-14")))

pandas_df = pd.read_csv('Data\covid19.csv')


pandas_df['dateRep'] = pandas_df['dateRep'].astype('datetime64[ns]')


mySchema = StructType([ StructField("dateRep", DateType(), True)\
                       ,StructField("day", IntegerType(), True)\
                       ,StructField("month", IntegerType(), True)\
                       ,StructField("year", IntegerType(), True)\
                        ,StructField("cases", IntegerType(), True)\
                        ,StructField("deaths", IntegerType(), True)\
                       ,StructField("countriesAndTerritories", StringType(), True)\
                       ,StructField("geoId", StringType(), True)\
                       ,StructField("iso_a3", StringType(), True)\
                       ,StructField("popData2019", FloatType(), True)\
                       ,StructField("continentExp", StringType(), True)])


s_df = sql_sc.createDataFrame(pandas_df,schema=mySchema)
s_df=s_df.withColumn("date",to_date(F.concat_ws("-","year","month","day")))
#s_df.show()
total= s_df.groupBy("iso_a3").agg(expr("sum(cases) as cases"),expr("sum(deaths) as deaths"))

## Total cases plot
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
country_shapes = world[['geometry', 'iso_a3']]
country_shapes = country_shapes.merge(total.toPandas(), on='iso_a3')
country_shapes.plot(column='cases',legend=True)
#plt.show()
country_shapes.plot(column='deaths',legend=True)
#plt.show()
#max_min= s_df.withColumn("date",F.col("date")).groupBy("iso_a3").agg(expr("max(date) as max"),expr("min(date) as min"))
#max_min.show()
#today = datetime(2020,6,23)
#new_date =today-timedelta(14)
#dates = (today,  new_date)
#between_dates= s_df.where(F.col('date').between(*dates)).show(truncate=False)

s_country=s_df.filter("iso_a3=='IND'").sort('date', ascending=True)
non_zero = s_country.filter("cases != 0").first()
min_date = non_zero['date']
min_max_con =s_country.withColumn("date",F.col("date")).groupBy("iso_a3").agg(expr("max(date) as max"),expr("min(date) as min"))
min_max_con.show()
max_date = min_max_con.first()[1]
dates= (min_date , max_date )
s_country= s_country.filter(F.col('date').between(*dates))
s_country.show()
s_country = s_country.withColumn("dateIndex",
              datediff(to_date("date","yyyy-MM-dd"),to_date(F.lit(min_date)))).sort('date', ascending=True)
#s_df=s_df.withColumn("idx",F.monotonically_increasing_id())
new_date =max_date-timedelta(1)
s_country.show()
s_df1=s_country.filter("iso_a3=='IND'").filter((F.col('date') != max_date) & (F.col('date') != new_date)).sort('date', ascending=False)
s_df2=s_country.filter("iso_a3=='IND'").filter((F.col('date') == max_date) | (F.col('date') == new_date))
s_df1.show()
s_df2.show()
vectorAssembler= VectorAssembler(inputCols=['dateIndex'],outputCol='features')
data= vectorAssembler.transform(s_df1)
train= data.select(['features','cases','dateIndex'])

data= vectorAssembler.transform(s_df2)
test= data.select(['features','cases','dateIndex'])
data.show(3)
df= data.toPandas()
#df.plot()
#plt.show()
lr = LinearRegression(featuresCol='features',labelCol='cases')
lr_model = lr.fit(train)

print(lr_model.coefficients)
print(lr_model.intercept)

lr_p = lr_model.transform(test)
lr_p.select("prediction","cases","features").show()
plt.figure(300)
x1 = train.toPandas()['dateIndex'].values.tolist()
y1 = train.toPandas()['cases'].values.tolist()
plt.scatter(x1, y1, color='blue', s=30)
x2 = lr_p.toPandas()['dateIndex'].astype(int).values.tolist()
y2 = lr_p.toPandas()['prediction'].values.tolist()
plt.scatter(x2, y2, color='red', s=30)
plt.show()


