

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
  Here we consider a cross-validator approach.
  The dataset is splitted into a set of non-overlapping randomly partitioned folds which are used as separate training and test datasets
  They are used to compute the sum of squared error (SSE) for a range of value of k
*/

object KMeans03 {

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
    
   // Cross-Validator
   val nbFolds = 2
   val splits = MLUtils.kFold(data, nbFolds, 123L)
//   println(splits)
//   println(splits.length)
   var splitTrainingCost: Map[Int, Seq[(Int, Double)]] = Map()
   var splitValidationCost: Map[Int, Seq[(Int, Double)]] = Map()

   (0 to splits.length-1 by 1).foreach(index => {
      splits(index)._1.persist()
      splits(index)._2.persist()
      val trainingCost = new KMeansTask(splits(index)._1, splits(index)._1, 4 to 20 by 1).computeElbowMethod()
      val validationCost = new KMeansTask(splits(index)._1, splits(index)._2, 4 to 20 by 1).computeElbowMethod()
      splitTrainingCost ++= Map(index -> trainingCost)
      splitValidationCost ++= Map(index -> validationCost)
      splits(index)._1.unpersist()
      splits(index)._2.unpersist()}) 

   splitTrainingCost = splitTrainingCost
                        .map(x => x._2)
                        .reduce(_ ++ _)
                        .groupBy(x => x._1)
   val trainingCost = splitTrainingCost.map(x => (x._1, x._2.map(y => y._2).sum / nbFolds.toDouble)).toList
   spark.createDataFrame(trainingCost).toDF("k", "cost")
        .write.mode(SaveMode.Overwrite).parquet("target/data/crossvalidation/trainingCost")  

   splitValidationCost = splitValidationCost
                        .map(x => x._2)
                        .reduce(_ ++ _)
                        .groupBy(x => x._1)
   val validationCost = splitTrainingCost.map(x => (x._1, x._2.map(y => y._2).sum / nbFolds.toDouble)).toList
   spark.createDataFrame(validationCost).toDF("k", "cost")
        .write.mode(SaveMode.Overwrite).parquet("target/data/crossvalidation/validationCost")  

    spark.stop()
  }

}
// scalastyle:on println
