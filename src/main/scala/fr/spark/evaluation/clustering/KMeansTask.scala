package fr.spark.evaluation.clustering

import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans}
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector

class KMeansTask(val featuresColumn: String, predictionColumn: String) extends ClusteringFactory {

  var kMeans: KMeans = _
  var model: KMeansModel = _

  def defineModel(k: Int): KMeansTask = {
  kMeans = new KMeans()
    .setK(k)
    .setFeaturesCol(featuresColumn)
    .setPredictionCol(predictionColumn)
    .setInitMode(MLlibKMeans.K_MEANS_PARALLEL)
  this
  }

  override def fit(data: DataFrame): KMeansTask = {
    model = kMeans.fit(data)
    this
  }

  override def transform(data: DataFrame): DataFrame = model.transform(data)

  override def computeCost(data: DataFrame): Double = model.computeCost(data)

  override def clusterCenters(): Array[Vector] = {
    model.clusterCenters
  }
}
