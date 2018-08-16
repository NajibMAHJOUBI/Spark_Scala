package fr.spark.evaluation.criterium

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.linalg.{Vector, DenseVector => MLDV}
import org.apache.spark.sql.{DataFrame, SparkSession}

// percentage of variance explained: ratio of hte between-group variance to the total variance,
// also known as an F-Test

class ExplainedVarianceTask(val featureColumn: String, val predictionColumn: String){

  def run(spark: SparkSession, data: DataFrame, withinVariance: Double, clusterCenters: Array[Vector]): Double = {
    computeFTest(spark, data, withinVariance, clusterCenters)
  }

  def getNumberOfObservation(data: DataFrame): Long = {data.count()}

  def computeDataCenter(spark: SparkSession, data: DataFrame): BDV[Double] = {
    val featureColumnBroadcast = spark.sparkContext.broadcast(featureColumn)
    (1.0/getNumberOfObservation(data).toDouble) * data
      .rdd.map(row => row.getAs[MLDV](row.fieldIndex(featureColumnBroadcast.value)).toArray)
      .map(p => new BDV(p))
      .reduce(_ + _)
  }

  def getNumberOfObservationsByCluster(spark: SparkSession, data: DataFrame): Map[Int, Long] = {
    val predictionColumnBroadcast = spark.sparkContext.broadcast(predictionColumn)
    data
      .groupBy(predictionColumn).count()
      .rdd.map(row => (row.getInt(row.fieldIndex(predictionColumnBroadcast.value)),
      row.getLong(row.fieldIndex("count"))))
      .collectAsMap().toMap
  }

  def squaredDistance(vec1: BDV[Double], vec2: BDV[Double]): Double =  {
    val diff = vec1 - vec2
    diff.dot(diff)
  }

  def computeBetweenVariance(spark: SparkSession, data: DataFrame, clusterCenters: Array[Vector]): Double = {
    val centerData: BDV[Double] = computeDataCenter(spark, data)
    val clustersCountMap = getNumberOfObservationsByCluster(spark: SparkSession, data: DataFrame)
    var betweenVariance = 0.0
    (0 until clusterCenters.length).foreach(index => {
      val diffVector = new BDV(clusterCenters(index).toArray) - centerData
      betweenVariance = betweenVariance + clustersCountMap(index) * squaredDistance(new BDV(clusterCenters(index).toArray), centerData)
    })
    betweenVariance
  }

  def computeTotalVariance(betweenVariance: Double, withinVariance: Double): Double = {
    betweenVariance + withinVariance
  }

  def computeFTest(spark: SparkSession, data: DataFrame, withinVariance: Double, clusterCenters: Array[Vector]): Double = {
    val betweenVariance =  computeBetweenVariance(spark, data, clusterCenters)
    val totalVariance = computeTotalVariance(betweenVariance, withinVariance)
    betweenVariance / totalVariance
  }

}
