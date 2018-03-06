package fr.spark.clustering.kmeans

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.K_MEANS_PARALLEL
import breeze.linalg.DenseVector

class ExplainedVarianceTask(data: RDD[Vector]) {

  var dataBreeze : RDD[DenseVector[Double]] = _
  var dataCenter: DenseVector[Double] = _
  var totalVariance: Double = _
  var betweenVariance:Double = _
  var withinVariance: Double = _

  def prepareData(): ExplainedVarianceTask = {
    dataBreeze = data.map(x => DenseVector(x.toDense.values))
    this
  }

  def computeDataCenter(): ExplainedVarianceTask = {
    dataCenter = dataBreeze.reduce(_ + _) / dataBreeze.count().toDouble
    this}

  def computeTotalVariance(dataCenter: DenseVector[Double]): ExplainedVarianceTask = {
    totalVariance = dataBreeze.map(x => {val diff = x - dataCenter; diff.dot(diff)}).reduce(_ + _)
    this}

  def computeBetweenVariance(kmeans: KMeansModel): ExplainedVarianceTask = {
    val clusterCenters = kmeans.clusterCenters.map(x => DenseVector(x.toDense.values))
    val predictionCount = kmeans.predict(data).countByValue
    val centerSquaredDistance = clusterCenters.map(x => squaredDistance(x, dataCenter))
    betweenVariance = predictionCount.map(x => x._2.toDouble * centerSquaredDistance(x._1)).sum
    this} 

   def computeWithinVariance(): ExplainedVarianceTask = {
     withinVariance = totalVariance - betweenVariance
     this}

    def computeExplainedVariance(): Double = {betweenVariance / totalVariance}

    def computeChIndex(nbData: Integer, nbCluster: Integer): Double = {
      ((nbData - nbCluster).toDouble * betweenVariance) / ((nbCluster - 1).toDouble * withinVariance)}

    val squaredDistance = (u: DenseVector[Double], v: DenseVector[Double]) => {
      val diff = u - v
      diff.dot(diff)}


  }


