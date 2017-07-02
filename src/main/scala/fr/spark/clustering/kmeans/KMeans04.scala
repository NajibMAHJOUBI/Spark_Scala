

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, DataFrame, SaveMode}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.DenseVector
import fr.spark.clustering.kmeans.KMeansTask
import fr.spark.clustering.kmeans.ExplainedVariance

/**
  explained variance
*/

object KMeans04 {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("KMeans")
      .getOrCreate()


    val path = "data/clustering"
    val data = spark.read.parquet(path)
                .rdd
                .map(p => Vectors.dense(p.getDouble(p.fieldIndex("x")), p.getDouble(p.fieldIndex("y"))))
                .persist()

    val kmeans = new KMeansTask().computeKMeansModel(data, 16)

    val explainedVariance = new ExplainedVariance(kmeans, data).computeExplainedVariance()
    println(s"explainedVariance = $explainedVariance")
/**
    val dataBreeze = data.map(x => DenseVector(x.toDense.values))  
                               .persist()
    val dataCount = dataBreeze.count()
    val dataCenter = dataBreeze.reduce(_ + _) / dataCount.toDouble

    println(dataCenter)
     
    val kmeans = new KMeansTask().computeKMeansModel(data, 16)
    val clusterCenters = kmeans.clusterCenters.map(x => DenseVector(x.toDense.values))
    val predictionCount = kmeans.predict(data).countByValue
    println(predictionCount)
    val centerSquaredDistance = clusterCenters.map(x => squaredDistance(x, dataCenter))
    val betweenVariance = predictionCount.map(x => x._2.toDouble * centerSquaredDistance(x._1)).sum
    val totalVariance = dataBreeze.map(x => squaredDistance(x, dataCenter)).reduce(_ + _)
    val explainedVariance = betweenVariance / totalVariance

    println(s"betweenVariance = $betweenVariance")
    println(s"totalVariance = $totalVariance")
    println(s"explainedVariance = $explainedVariance")
    */
    spark.stop()
  }


}
// scalastyle:on println
