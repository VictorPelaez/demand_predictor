from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.regression import LinearRegression

import pandas as pd
import numpy as np
import os.path
import time
import datetime
import urllib

# --------------
link_prev = 'https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/prevProgPeninsula?callback=angular.callbacks._1&curva=Generacion&fecha='
link_gen = 'https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/demandaGeneracionPeninsula?callback=angular.callbacks._2&curva=DEMANDA&fecha='

def parseurl_todf(link, day):
  f = urllib.urlopen(link+day)           
  myfile = f.readline()  
  myfile= myfile[51:-4]
  file = myfile.split(",{")[1:]
  file = ['{'+l for l in file]
  json_obj = sc.parallelize(file)
  
  return sqlContext.read.json(json_obj)
  
 def create_days_vector(m, dayst, dayend):
  days =[]
  for day in range(dayst,dayend):
    day = str(day)
    if len(day)==1: day='0'+day;
    days.append('2017-'+ m + '-'+ str(day))
  return days

days = create_days_vector('02', 14, 19)
print days 

l =0
for day in days:
  df_prev_temp = parseurl_todf(link_prev, day)
  if l ==0: df_prev = df_prev_temp; l=1;
  else: df_prev = df_prev.unionAll(df_prev_temp); df_prev=df_prev.distinct()

print df_prev.count()     
l =0
for day in days:
  df_gen_temp = parseurl_todf(link_gen, day)
  if l ==0: df_gen = df_gen_temp; l=1;
  else: df_gen = df_gen.unionAll(df_gen_temp); df_gen=df_gen.distinct()  
# --------------

df = df_prev.join(df_gen, df_prev.ts==df_gen.ts).drop(df_gen.ts).sort(col("ts"))
func =  udf (lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M'), DateType())
df = df.select('*', unix_timestamp(func(col('ts'))).alias("id"))

w = Window().partitionBy().orderBy(col("id"))
df_featured = df.select("*", hour(col("ts")).alias("hour"), date_format(col("ts"), 'EEEE').alias("weekday"), lag("pro").over(w).alias("pro_lag1"), lag("pre").over(w).alias("pre_lag1"))

df_featured = df_featured.select( col("dem").alias("label"), col("ts"),col("id"), col("hour"),col("weekday"),col("pro_lag1"),col("pre_lag1"), col("pro"),col("pre")).filter(col("pro_lag1")>0)
df_featured.printSchema()



training_seti = df_featured.select(col("pro_lag1"), col("pre_lag1"),col("hour"), col("ts"), col("label"))

vectorizer = VectorAssembler()
vectorizer.setInputCols(["pro_lag1", "pre_lag1", "hour"])
vectorizer.setOutputCol("features")

# Let's initialize our linear regression learner
lr = LinearRegression()

lr.setPredictionCol("prediction")\
  .setMaxIter(100)\
  .setRegParam(0.1)

# We will use the new spark.ml pipeline API. If you have worked with scikit-learn this will be very familiar.
lrPipeline = Pipeline()
lrPipeline.setStages([vectorizer,lr])

lrModel = lrPipeline.fit(training_seti)

predicted_df = lrModel.transform(training_seti)
# display(predicted_df)

test_seti = df_featured.select(col("pro").alias("pro_lag1"), col("pre").alias("pre_lag1"),col("hour"), col("ts"))
predicted_test_df = lrModel.transform(test_seti)
