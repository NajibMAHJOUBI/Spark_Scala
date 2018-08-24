package fr.spark.evaluation.criterium

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, SparkSession}

class ChIndexMethodTask(val featureColumn: String, val predictionColumn: String) {

  def run(spark: SparkSession, data: DataFrame, withinVariance: Double, clusterCenters: Array[Vector]): Double = {
    computeCHIndex(spark, data, withinVariance, clusterCenters)
  }

  def computeCHIndex(spark: SparkSession, data: DataFrame, withinVariance: Double, clusterCenters: Array[Vector]): Double = {
    val numberOfObservation = CriteriumObject.getNumberOfObservation(data)
    val numberOfCluster = CriteriumObject.getNumberOfCluster(data, predictionColumn)
    val betweenVariance =  CriteriumObject.computeBetweenVariance(spark, data, clusterCenters, featureColumn, predictionColumn)
    (betweenVariance / (numberOfCluster.toDouble - 1.0))/ (withinVariance / (numberOfObservation.toDouble - numberOfCluster.toDouble))
  }

}
