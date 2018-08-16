package fr.spark.evaluation.clustering

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.linalg.Vector

trait ClusteringFactory {

  def defineModel(k: Int): ClusteringFactory

  def fit(data: DataFrame): ClusteringFactory

  def transform(data: DataFrame): DataFrame

  def computeCost(data: DataFrame): Double

  def clusterCenters(): Array[Vector]
}