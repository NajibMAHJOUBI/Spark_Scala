package fr.spark.evaluation.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.junit.{After, Before, Test}

class DefineFeaturesTaskTest {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("define features test suite")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testDefineFeatures(): Unit = {
    val data = new LoadDataTask("src/test/scala/resources/").run(spark, "clustering")
    val featuresDf = new DefineFeaturesTask(Array("x", "y"), "features").run(spark, data)

    assert(featuresDf.isInstanceOf[DataFrame])
    assert(featuresDf.count() == data.count())
    assert(featuresDf.columns.contains("features"))
    assert(featuresDf.schema.fields(featuresDf.schema.fieldIndex("features")).dataType.typeName == "vector")
  }

  @After def afterAll() {
    spark.stop()
  }

}



