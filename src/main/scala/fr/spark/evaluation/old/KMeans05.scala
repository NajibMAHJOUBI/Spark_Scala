

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
  CH-index
*/

object KMeans05 {

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
    
    val nbData = data.count()
    val explainedVarianceTask = new ExplainedVarianceTask(data)
    explainedVarianceTask.prepareData()
    explainedVarianceTask.computeDataCenter()
    explainedVarianceTask.computeTotalVariance(explainedVarianceTask.dataCenter)
    println(explainedVarianceTask.totalVariance)
    
    
    var variance: List[(Int, Double)] = List()
    (4 to 30 by 1).foreach(nbCluster => {
      val kmeans = new KMeansTask().computeKMeansModel(data, nbCluster) 
      //println(kmeans.computeCost(data))
      explainedVarianceTask.computeBetweenVariance(kmeans)
      explainedVarianceTask.computeWithinVariance()
      variance = variance ++ List((nbCluster, explainedVarianceTask.computeChIndex(nbData.toInt, nbCluster)))
    })

     spark.createDataFrame(variance).toDF("k", "ch_index")
          .write.mode(SaveMode.Overwrite).parquet("target/data/chIndex")

    spark.stop()
  }


}
// scalastyle:on println
