

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, DataFrame, SaveMode}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

import fr.spark.clustering.kmeans.KMeansTask

/**
  Elbow method
  All the dataset is used to compute the sum of squared error (SSE)
  for a range of value of k
*/

object KMeans01 {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("KMeans")
      .getOrCreate()


    val path = "data/clustering"
    var data = spark.read.parquet(path)
                .rdd
                .map(p => Vectors.dense(p.getDouble(p.fieldIndex("x")), p.getDouble(p.fieldIndex("y")))) 
                .persist()
    
   
   val kmeans = new KMeansTask(data, data, 4 to 20 by 1)
   val elbowCost = kmeans.computeElbowMethod()
   spark.createDataFrame(elbowCost).toDF("k", "cost").write.mode(SaveMode.Overwrite).parquet("target/data/elbowcost") 

    spark.stop()
  }

}
// scalastyle:on println
