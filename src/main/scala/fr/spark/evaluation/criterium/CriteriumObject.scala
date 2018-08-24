package fr.spark.evaluation.criterium


import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.linalg.{Vector, DenseVector => MLDV}


object CriteriumObject {

  def getNumberOfObservation(data: DataFrame): Long = {data.count()}

  def getNumberOfCluster(data: DataFrame, predictionColumn: String): Long = {
    data.select(predictionColumn)distinct()count()
  }

  def computeDataCenter(spark: SparkSession, data: DataFrame, featureColumn: String): BDV[Double] = {
    val featureColumnBroadcast = spark.sparkContext.broadcast(featureColumn)
    (1.0/getNumberOfObservation(data).toDouble) * data
      .rdd.map(row => row.getAs[MLDV](row.fieldIndex(featureColumnBroadcast.value)).toArray)
      .map(p => new BDV(p))
      .reduce(_ + _)
  }

  def getNumberOfObservationsByCluster(spark: SparkSession, data: DataFrame, predictionColumn: String): Map[Int, Long] = {
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

  def computeBetweenVariance(spark: SparkSession, data: DataFrame, clusterCenters: Array[Vector], featureColumn: String, predictionColumn: String): Double = {
    val centerData: BDV[Double] = computeDataCenter(spark, data, featureColumn)
    val clustersCountMap = getNumberOfObservationsByCluster(spark, data, predictionColumn)
    var betweenVariance = 0.0
    (0 until clusterCenters.length).foreach(index => {
      betweenVariance = betweenVariance + clustersCountMap(index) * squaredDistance(new BDV(clusterCenters(index).toArray), centerData)
    })
    betweenVariance
  }

  def computeTotalVariance(betweenVariance: Double, withinVariance: Double): Double = {
    betweenVariance + withinVariance
  }

}
