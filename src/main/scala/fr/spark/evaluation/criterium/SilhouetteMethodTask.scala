package fr.spark.evaluation.criterium

import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.sql.DataFrame

class SilhouetteMethodTask(val featureColumn: String, val predictionColumn: String) {

  def run(data: DataFrame): Double = {
    defineEvaluator().evaluate(data)
  }

  def defineEvaluator(): ClusteringEvaluator = {
    new ClusteringEvaluator()
      .setFeaturesCol(featureColumn)
      .setPredictionCol(predictionColumn)
      .setMetricName("silhouette")
  }

}
