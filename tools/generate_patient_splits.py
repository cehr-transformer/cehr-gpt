import os
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for downloading OMOP tables')

    parser.add_argument('--input_folder',
                        dest='input_folder',
                        action='store',
                        required=True)
    parser.add_argument('--output_folder',
                        dest='output_folder',
                        action='store',
                        required=True)
    ARGS = parser.parse_args()
    spark = SparkSession.builder.appName("Generate Patient Splits").getOrCreate()
    person = spark.read.parquet(os.path.join(ARGS.input_folder, "person"))
    train_person, test_person = person.randomSplit([0.8, 0.2], seed=42)
    train_person = train_person.select("person_id").withColumn("split", lit("train"))
    test_person = test_person.select("person_id").withColumn("split", lit("test"))
    patient_splits = train_person.unionByName(test_person)
    patient_splits.write.mode("overwrite").parquet(os.path.join(ARGS.output_folder, "patient_splits"))
