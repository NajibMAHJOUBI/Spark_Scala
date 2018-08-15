package fr.spark.evaluation.clustering

import org.apache.spark.sql.DataFrame

trait ClusteringFactory {

  def defineModel(k: Int): ClusteringFactory

  def fit(data: DataFrame): ClusteringFactory

  def transform(data: DataFrame): DataFrame

  def computeCost(data: DataFrame): Double

}