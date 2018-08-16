package fr.spark.evaluation.utils

import org.apache.log4j.{Level, LogManager}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.junit.{After, Before, Test}

class LoadDataTaskTest {

  private var spark: SparkSession = _

  @Before def beforeAll() {
    spark = SparkSession
      .builder
      .master("local")
      .appName("KMeans test suite")
      .getOrCreate()

    val log = LogManager.getRootLogger
    log.setLevel(Level.WARN)
  }

  @Test def testLoadData(): Unit = {
    val data = new LoadDataTask("src/test/scala/resources/").run(spark, "clustering")
    assert(data.isInstanceOf[DataFrame])
    assert(data.count() == 2)
    assert(data.columns.contains("x"))
    assert(data.columns.contains("y"))
  }

  @After def afterAll() {
    spark.stop()
  }

}



