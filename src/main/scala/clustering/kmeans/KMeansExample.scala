

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SparkSession, DataFrame, SaveMode}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors




object KMeansExample {

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

   // Elbow method
   val elbowCost = computeElbowMethod(spark, data, data)
   spark.createDataFrame(elbowCost).toDF("k", "cost").write.mode(SaveMode.Overwrite).parquet("target/data/elbowcost")

   // Train validation
   val trainRatio = 0.75
   val Array(training, validation) = data.randomSplit(Array(trainRatio, 1.0 - trainRatio), 123L)
   training.persist()
   validation.persist()
   val trainingTrainingCost = computeElbowMethod(spark, training, training)
   val trainingValidationCost = computeElbowMethod(spark, training, validation)
   spark.createDataFrame(trainingTrainingCost).toDF("k", "cost").write.mode(SaveMode.Overwrite).parquet("target/data/trainvalidation/elbowcost_trainingTraining")
   spark.createDataFrame(trainingValidationCost).toDF("k", "cost").write.mode(SaveMode.Overwrite).parquet("target/data/trainvalidation/elbowcost_trainingValidation")
   training.unpersist()
   validation.unpersist()

   // Cross-Validator
   val splits = MLUtils.kFold(data, 5, 123L)
   println(splits)
   println(splits.length)
   var splitTrainingCost: Map[Int, Seq[(Int, Double)]] = Map()
   var splitValidationCost: Map[Int, Seq[(Int, Double)]] = Map()

   (0 to splits.length-1 by 1).foreach(index => {
      splits(index)._1.persist()
      splits(index)._2.persist()
      val trainingCost = computeElbowMethod(spark, splits(index)._1, splits(index)._1)
      val validationCost = computeElbowMethod(spark, splits(index)._1, splits(index)._2)
      splitTrainingCost ++= Map(index -> trainingCost)
      splitValidationCost ++= Map(index -> validationCost)
      splits(index)._1.unpersist()
      splits(index)._2.unpersist()
   }) 

    spark.stop()
  }

  def computeKMeansModel(data: RDD[Vector], nbCluster: Int): KMeansModel = {
     new KMeans()
      .setK(nbCluster)
      .setMaxIterations(20)
      .setInitializationSteps(10)
      .setInitializationMode(K_MEANS_PARALLEL)
      .run(data)}

  def computeElbowMethod(spark: SparkSession, training: RDD[Vector], validation: RDD[Vector]): Seq[(Int, Double)] = {
    var resultElbow: List[(Int, Double)] = List()
    (4 to 20 by 1).foreach(nbCluster => {
         val cost: Double = computeKMeansModel(training, nbCluster).computeCost(validation)
         resultElbow = resultElbow ++ List((nbCluster, cost))})
    resultElbow
}

}
// scalastyle:on println
