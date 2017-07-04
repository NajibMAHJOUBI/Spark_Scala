

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, DataFrame, SaveMode}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.DenseVector
import fr.spark.clustering.kmeans.KMeansTask
import fr.spark.clustering.kmeans.ExplainedVarianceTask

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
    
    var variance: List[(Int, Double)] = List()
    (4 to 20 by 1).foreach(nbCluster => {
      val kmeans = new KMeansTask().computeKMeansModel(data, nbCluster)
      val explainedVarianceTask = new ExplainedVarianceTask(kmeans, data)
      explainedVarianceTask.prepareData()
      explainedVarianceTask.computeDataCenter()
      explainedVarianceTask.computeTotalVariance(explainedVarianceTask.dataCenter)
      explainedVarianceTask.computeBetweenVariance()   
      val explainedVariance = explainedVarianceTask.computeExplainedVariance()
      variance = variance ++ List((nbCluster, explainedVariance))
    })
   
     spark.createDataFrame(variance).toDF("k", "explained_variance")
          .write.mode(SaveMode.Overwrite).parquet("target/data/explainedVariance")

    spark.stop()
  }


}
// scalastyle:on println
