package fr.spark.evaluation.suiteTest

import fr.spark.evaluation.clustering.{BisectingKMeansTask, KMeansTask}
import fr.spark.evaluation.utils.{DefineFeaturesTask, LoadDataTask}
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.SparkSession
import java.io._

import fr.spark.evaluation.criterium.{ExplainedVarianceTask, SilhouetteMethodTask}

object testKMeans {

  def main(arguments: Array[String]): Unit = {
    val spark = SparkSession.builder.master("local").appName("Elbow method - cost function").getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    val featureColumn= "features"
    val predictionColumn = "prediction"
    val data = new LoadDataTask("data").run(spark, "clustering")
    val features = new DefineFeaturesTask(Array("x", "y"), featureColumn).run(spark, data)

    val kMeans = new KMeansTask(featureColumn, predictionColumn)
    val explainedVariance = new ExplainedVarianceTask(featureColumn, predictionColumn)
    val silhouetteMethod = new SilhouetteMethodTask(featureColumn, predictionColumn)

    var costKMeansList: List[(Int, Double)] = List()
    var explainedVarianceList: List[(Int, Double)] = List()
    var silhouetteList: List[(Int, Double)] = List()

    (2 to 20).foreach(k => {
      kMeans.defineModel(k)
      kMeans.fit(features)
      val transform = kMeans.transform(features)
      costKMeansList = costKMeansList ++ List((k, kMeans.computeCost(features)))
      explainedVarianceList = explainedVarianceList ++ List((k, explainedVariance.run(spark, transform, kMeans.computeCost(features), kMeans.clusterCenters())))
      silhouetteList = silhouetteList ++ List((k, silhouetteMethod.run(transform)))
    })

    saveCost(costKMeansList, "submission/kMeans/elbow.csv")
    saveCost(explainedVarianceList, "submission/kMeans/explainedVariance.csv")
    saveCost(silhouetteList, "submission/kMeans/silhouetteMethod.csv")
  }

  def saveCost(resultList: List[(Int, Double)], savePath: String): Unit = {
    val pw = new PrintWriter(new File(savePath))
    pw.write("k;cost\n")
    resultList.foreach(result => pw.write(s"${result._1};${result._2}\n"))
    pw.close
  }
}
