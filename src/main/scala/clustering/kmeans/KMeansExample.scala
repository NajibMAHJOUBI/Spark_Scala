
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.clustering.KMeans.{K_MEANS_PARALLEL, RANDOM}
import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.SaveMode



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
   elbowCost.write.mode(SaveMode.Overwrite).parquet("target/data/elbowcost")

   // Train validation
   val trainRatio = 0.75
   val Array(training, validation) = data.randomSplit(Array(trainRatio, 1.0 - trainRatio), 123L)
   training.persist()
   validation.persist()
   //training.write.mode(SaveMode.Overwrite).parquet("target/data/trainvalidation/training")
   //validation.write.mode(SaveMode.Overwrite).parquet("target/data/trainvalidation/validation")
   val trainValidationCost = computeElbowMethod(spark, training, validation)
   trainValidationCost.write.mode(SaveMode.Overwrite).parquet("target/data/trainvalidation/elbowcost")

    spark.stop()
  }

  def computeKMeansModel(data: RDD[Vector], nbCluster: Int): KMeansModel = {
     new KMeans()
      .setK(nbCluster)
      .setMaxIterations(20)
      .setInitializationSteps(10)
      .setInitializationMode(K_MEANS_PARALLEL)
      .run(data)}

  def computeElbowMethod(spark: SparkSession, training: RDD[Vector], validation: RDD[Vector]): DataFrame = {
    var resultElbow: List[(Int, Double)] = List()
    (4 to 20 by 1).foreach(nbCluster => {
         val cost: Double = computeKMeansModel(training, nbCluster).computeCost(validation)
         resultElbow = resultElbow ++ List((nbCluster, cost))})
    resultElbow
    spark.createDataFrame(resultElbow).toDF("k", "cost")
}

}
// scalastyle:on println
