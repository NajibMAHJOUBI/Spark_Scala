package fr.spark.evaluation.utils

import org.apache.spark.sql.{DataFrame, SparkSession}

class LoadDataTask (val path: String) {

  def run(spark: SparkSession, subPath: String): DataFrame = {
    spark.read.parquet(s"$path/$subPath")
  }

}
