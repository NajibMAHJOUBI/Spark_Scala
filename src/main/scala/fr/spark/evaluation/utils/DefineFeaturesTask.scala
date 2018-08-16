package fr.spark.evaluation.utils

import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.WrappedArray


class DefineFeaturesTask(val valuesColumn: Array[String], val featureColumn: String) {

  def run(spark: SparkSession, data: DataFrame): DataFrame = {
    defineFeatures(spark, data)
  }

  def defineValues(spark: SparkSession, data: DataFrame): DataFrame = {
    val valuesBroadcast = spark.sparkContext.broadcast(valuesColumn)
    val rdd = data.rdd.map(p => valuesBroadcast.value.map(label => p.getDouble(p.fieldIndex(label))).toList)
    spark.createDataFrame(rdd).toDF("values")
  }

  def defineFeatures(spark: SparkSession, data: DataFrame): DataFrame = {
    val values = defineValues(spark, data)
    val getDenseVector = udf((values: WrappedArray[Double]) => UtilsObject.defineDenseVector(values))
    values.withColumn(featureColumn, getDenseVector(col("values"))).drop("values")
  }

  }


