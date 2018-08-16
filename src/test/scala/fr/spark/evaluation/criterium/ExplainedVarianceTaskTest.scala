package fr.spark.evaluation.criterium

import breeze.linalg.{DenseVector => BDV}
import fr.spark.evaluation.clustering.KMeansTask
import fr.spark.evaluation.utils.LoadDataTask
import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

class ExplainedVarianceTaskTest {

  private var spark: SparkSession = _
  private var data: DataFrame = _
  private val featuresColumn: String = "features"
  private val predictionColumn: String = "prediction"

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("explained variance test suite")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)

    data = new LoadDataTask("src/test/scala/resources/").run(spark, "features")

  }

  @Test def testNumberOfObservation(): Unit = {
    val explainedVariance = new ExplainedVarianceTask(featuresColumn, predictionColumn)
    val count = explainedVariance.getNumberOfObservation(data)
    assert(count.isInstanceOf[Long])
    assert(count == data.count())
  }

  @Test def testDataCenters(): Unit = {
    val explainedVariance = new ExplainedVarianceTask(featuresColumn, predictionColumn)
    val center = explainedVariance.computeDataCenter(spark, data)

    assert(center.isInstanceOf[BDV[Double]])
  }

  @Test def testNumberOfObservationsByCluster(): Unit = {
    val kMeans = new KMeansTask(featuresColumn, predictionColumn)
    kMeans.defineModel(5)
    kMeans.fit(data)
    val transform = kMeans.transform(data)

    val explainedVariance = new ExplainedVarianceTask(featuresColumn, predictionColumn)
    val clustersCountMap = explainedVariance.getNumberOfObservationsByCluster(spark, transform)
    assert(clustersCountMap.isInstanceOf[Map[Int, Long]])
  }

  @Test def testBetweenVariance(): Unit = {
    val kMeans = new KMeansTask(featuresColumn, predictionColumn)
    kMeans.defineModel(5)
    kMeans.fit(data)
    val transform = kMeans.transform(data)

    val explainedVariance = new ExplainedVarianceTask(featuresColumn, predictionColumn)
    val centerData = explainedVariance.computeDataCenter(spark, data)
    val clustersCountMap = explainedVariance.getNumberOfObservationsByCluster(spark, transform)
    val clusterCenter = kMeans.clusterCenters()

    println(centerData)
    println(clustersCountMap)
    clusterCenter.foreach(println)

    clusterCenter.map(center => new BDV(center.toArray) - centerData).foreach(println)

    var betweenVariance = 0.0
    (0 until clusterCenter.length).foreach(index => {
      val diffVector = new BDV(clusterCenter(index).toArray) - centerData
      betweenVariance = betweenVariance + clustersCountMap(index) * diffVector.dot(diffVector)
    })
    println(s"betweenVariance: $betweenVariance")
  }

  @After def afterAll() {
    spark.stop()
  }

}



