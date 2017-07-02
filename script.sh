sbt clean package
spark-submit --class KMeans01 target/scala-2.11/sparkscala_2.11-1.0.jar 
spark-submit --class KMeans02 target/scala-2.11/sparkscala_2.11-1.0.jar 
spark-submit --class KMeans03 target/scala-2.11/sparkscala_2.11-1.0.jar 

