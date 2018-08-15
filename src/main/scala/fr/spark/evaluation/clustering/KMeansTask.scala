package fr.spark.evaluation.clustering

import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans}
import org.apache.spark.sql.DataFrame

class KMeansTask(val featuresColumn: String, predictionColumn: String) {

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

  def fit(data: DataFrame): KMeansTask = {
    model = kMeans.fit(data)
    this
  }

  def transform(data: DataFrame): DataFrame = {
    model.transform(data)
  }

}
