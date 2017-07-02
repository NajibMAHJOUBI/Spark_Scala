package fr.spark.clustering.kmeans

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.K_MEANS_PARALLEL

import scala.collection.immutable

class KMeansTask() {

  def computeKMeansModel(data: RDD[Vector], nbCluster: Int): KMeansModel = {
     new KMeans()
      .setK(nbCluster)
      .setMaxIterations(20)
      .setInitializationSteps(10)
      .setInitializationMode(K_MEANS_PARALLEL)
      .run(data)}

  def computeElbowMethod(training: RDD[Vector], validation: RDD[Vector], rangeK: immutable.Range): Seq[(Int, Double)] = {
    var resultElbow: List[(Int, Double)] = List()
    rangeK.foreach(nbCluster => {
         val cost: Double = computeKMeansModel(training, nbCluster).computeCost(validation)
         resultElbow = resultElbow ++ List((nbCluster, cost))})
    resultElbow}

  

}
