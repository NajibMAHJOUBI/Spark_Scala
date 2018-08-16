package fr.spark.evaluation.clustering

import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.DataFrame

class BisectingKMeansTask(val featuresColumn: String, predictionColumn: String) extends ClusteringFactory {

  var bisectingKMeans: BisectingKMeans = _
  var model: BisectingKMeansModel = _

  def defineModel(k: Int): BisectingKMeansTask = {
    bisectingKMeans = new BisectingKMeans()
      .setK(k)
      .setFeaturesCol(featuresColumn)
      .setPredictionCol(predictionColumn)
    this
  }

  def fit(data: DataFrame): BisectingKMeansTask = {
    model = bisectingKMeans.fit(data)
    this
  }

  def transform(data: DataFrame): DataFrame = {
    model.transform(data)
  }

  def computeCost(data: DataFrame): Double = {
    model.computeCost(data)
  }

  override def clusterCenters(): Array[Vector] = {
    model.clusterCenters
  }

}
