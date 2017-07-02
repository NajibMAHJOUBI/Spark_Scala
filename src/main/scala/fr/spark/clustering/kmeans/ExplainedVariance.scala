package fr.spark.clustering.kmeans

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.K_MEANS_PARALLEL
import breeze.linalg.DenseVector

class ExplainedVariance(kmeans: KMeansModel, data: RDD[Vector]) {


  def prepareData(data: RDD[Vector]): RDD[DenseVector[Double]] = {
    data.map(x => DenseVector(x.toDense.values))
  }

    /**
    def squaredDistance(u: DenseVector[Double], v: DenseVector[Double]): Double = {
      val diff = u - v
      diff.dot(diff)}
*/
    
   val squaredDistance = (u: DenseVector[Double], v: DenseVector[Double]) => {val diff = u - v
              diff.dot(diff)}

  def computeDataCenter(data: RDD[DenseVector[Double]]): DenseVector[Double] = {
    data.reduce(_ + _) / data.count().toDouble
  }

  def computeTotalVariance(data: RDD[DenseVector[Double]], dataCenter: DenseVector[Double]): Double = {
    data
     .map(x => {val diff = x - dataCenter 
                diff.dot(diff)})
     .reduce(_ + _)
  }

  def computeBetweenVariance(kmeans: KMeansModel, data: RDD[Vector], dataCenter: DenseVector[Double]): Double = {
    val clusterCenters = kmeans
                         .clusterCenters
                         .map(x => DenseVector(x.toDense.values))
    val predictionCount = kmeans
                          .predict(data).countByValue
    val centerSquaredDistance = clusterCenters
                                .map(x => squaredDistance(x, dataCenter))
    predictionCount.map(x => x._2.toDouble * centerSquaredDistance(x._1)).sum
  } 

  def computeExplainedVariance(): Double = {
    val dataBreeze = prepareData(data)
    val dataCenter = computeDataCenter(dataBreeze)
    val totalVariance = computeTotalVariance(dataBreeze, dataCenter)
    val betweenVariance = computeBetweenVariance(kmeans, data, dataCenter)
    betweenVariance / totalVariance
 }


  }


