

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
  The dataset is splitted in a training and validation set
  The clustering is computed on the training set
  The sum of squared error (SSE) is computed on the validation set for a range of value of k
*/

object KMeans02 {

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
    

   val trainRatio = 0.75
   val Array(training, validation) = data.randomSplit(Array(trainRatio, 1.0 - trainRatio), 123L)
   training.persist()
   validation.persist()

    
   val trainingCost = new KMeansTask().computeElbowMethod(training, training, 4 to 20 by 1)
   val validationCost = new KMeansTask().computeElbowMethod(training, validation, 4 to 20 by 1)


   spark.createDataFrame(trainingCost).toDF("k", "cost")
       .write.mode(SaveMode.Overwrite).parquet("target/data/trainvalidation/trainingCost")

   spark.createDataFrame(validationCost).toDF("k", "cost")
       .write.mode(SaveMode.Overwrite).parquet("target/data/trainvalidation/validationCost")

   training.unpersist()
   validation.unpersist() 

    spark.stop()
  }

}
// scalastyle:on println
