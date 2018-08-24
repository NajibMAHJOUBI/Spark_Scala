package fr.spark.evaluation.criterium


import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.Vector

// percentage of variance explained: ratio of hte between-group variance to the total variance,
// also known as an F-Test

class ExplainedVarianceTask(val featureColumn: String, val predictionColumn: String){

  def run(spark: SparkSession, data: DataFrame, withinVariance: Double, clusterCenters: Array[Vector]): Double = {
    computeFTest(spark, data, withinVariance, clusterCenters)
  }

  def computeFTest(spark: SparkSession, data: DataFrame, withinVariance: Double, clusterCenters: Array[Vector]): Double = {
    val betweenVariance =  CriteriumObject.computeBetweenVariance(spark, data, clusterCenters, featureColumn, predictionColumn)
    val totalVariance = CriteriumObject.computeTotalVariance(betweenVariance, withinVariance)
    betweenVariance / totalVariance
  }

}
