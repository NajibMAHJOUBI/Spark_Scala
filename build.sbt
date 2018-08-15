name := "spark-clustering"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq("org.scalactic" %% "scalactic" % "3.0.5",
  "org.scalatest" %% "scalatest" % "3.0.5" % "test",
  "junit" % "junit" % "4.12" % Test,
  "org.apache.spark" % "spark-core_2.11" % "2.3.1",
  "org.apache.spark" % "spark-sql_2.11" % "2.3.1",
  "org.apache.spark" % "spark-mllib_2.11" % "2.3.1")
