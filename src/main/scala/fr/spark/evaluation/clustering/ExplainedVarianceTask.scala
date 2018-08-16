package fr.spark.evaluation.clustering

import org.apache.spark.sql.{DataFrame, SparkSession}

class ExplainedVarianceTask (val featureColumn: String, val predictionColumn: String){

  def run(spark: SparkSession, data: DataFrame): Unit = {

  }

  def computeDataCenter(data: DataFrame): Unit = {
//    data.rdd.map()
  }
}
