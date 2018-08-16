package fr.spark.evaluation.suiteTest

import fr.spark.evaluation.clustering.{BisectingKMeansTask, KMeansTask}
import fr.spark.evaluation.utils.{DefineFeaturesTask, LoadDataTask}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import java.io._

object testSuiteElbow {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Elbow method - cost function").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val featureColumn= "features"
    val predictionColumn = "prediction"
    val data = new LoadDataTask("data").run(spark, "clustering")
    val features = new DefineFeaturesTask(Array("x", "y"), featureColumn).run(spark, data)

    val kMeans = new KMeansTask(featureColumn, predictionColumn)
    val bisectingKMeans = new BisectingKMeansTask(featureColumn, predictionColumn)

    var costKMeansList: List[(Int, Double)] = List()
    var costBisectingKMeansList: List[(Int, Double)] = List()

    (2 to 20).foreach(k => {
      kMeans.defineModel(k)
      kMeans.fit(features)
      costKMeansList = costKMeansList ++ List((k, kMeans.computeCost(features)))

      bisectingKMeans.defineModel(k)
      bisectingKMeans.fit(features)
      costBisectingKMeansList = costBisectingKMeansList ++ List((k, bisectingKMeans.computeCost(features)))
    })
//    costKMeansList.foreach(println)
//    costBisectingKMeansList.foreach(println)

    saveCost(costKMeansList, "submission/elbow/kMeans/cost.csv")
    saveCost(costBisectingKMeansList, "submission/elbow/bisectingKMeans/cost.csv")
  }

  def saveCost(resultList: List[(Int, Double)], savePath: String): Unit = {
    val pw = new PrintWriter(new File(savePath))
    pw.write("k;cost\n")
    resultList.foreach(result => pw.write(s"${result._1};${result._2}\n"))
    pw.close
  }
}
