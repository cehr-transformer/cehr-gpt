import os
import argparse
import datetime

from pyspark.sql import SparkSession

import config.parameters
from utils.spark_utils import *

VISIT_OCCURRENCE = 'visit_occurrence'
PERSON = 'person'


def main(input_folder, output_folder, domain_table_list, date_filter,
         include_visit_type, is_new_patient_representation, exclude_visit_tokens,
         is_classic_bert, include_prolonged_stay, include_concept_list: bool):
    spark = SparkSession.builder.appName('Generate Bert Training Data').getOrCreate()
    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    visit_occurrence = visit_occurrence.select('visit_occurrence_id', 'visit_concept_id',
                                               'person_id')
    person = preprocess_domain_table(spark, input_folder, PERSON)
    person = person.select('person_id', F.coalesce('birth_datetime',
                                                   F.concat('year_of_birth', F.lit('-01-01')).cast(
                                                       'timestamp')).alias('birth_datetime'))
    visit_occurrence_person = visit_occurrence.join(person, 'person_id')

    patient_event = join_domain_tables(domain_tables)

    if include_concept_list and patient_event:
        # Filter out concepts
        qualified_concepts = preprocess_domain_table(
            spark,
            input_folder,
            config.parameters.qualified_concept_list_path
        ).select('standard_concept_id')

        patient_event = patient_event.join(
            qualified_concepts,
            'standard_concept_id'
        )

    patient_event = patient_event.join(visit_occurrence_person, 'visit_occurrence_id') \
        .select([patient_event[fieldName] for fieldName in patient_event.schema.fieldNames()] +
                ['visit_concept_id', 'birth_datetime']) \
        .withColumn('cohort_member_id', F.col('person_id')) \
        .withColumn('age', F.ceil(F.months_between(F.col('date'),
                                                   F.col("birth_datetime")) / F.lit(12)))

    if is_new_patient_representation:
        sequence_data = create_sequence_data_with_att(patient_event,
                                                      date_filter=date_filter,
                                                      include_visit_type=include_visit_type,
                                                      exclude_visit_tokens=exclude_visit_tokens)
    else:
        sequence_data = create_sequence_data(patient_event,
                                             date_filter=date_filter,
                                             include_visit_type=include_visit_type,
                                             classic_bert_seq=is_classic_bert)

    if include_prolonged_stay:
        udf = F.when(F.col('visit_concept_id').isin([9201, 262, 9203]),
                     F.coalesce((F.datediff('visit_end_date', 'visit_start_date') > 7).cast('int'),
                                F.lit(0))).otherwise(F.lit(0))
        visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
        visit_occurrence = visit_occurrence.withColumn('prolonged_length_stay', udf) \
            .select('person_id', 'prolonged_length_stay') \
            .withColumn('prolonged_length_stay',
                        F.max('prolonged_length_stay').over(W.partitionBy('person_id'))).distinct()
        sequence_data = sequence_data.join(visit_occurrence, 'person_id')

    sequence_data.write.mode('overwrite').parquet(os.path.join(output_folder,
                                                               config.parameters.parquet_data_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generate training data for Bert')
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the raw data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The path for your output_folder',
                        required=True)
    parser.add_argument('-tc',
                        '--domain_table_list',
                        dest='domain_table_list',
                        nargs='+',
                        action='store',
                        help='The list of domain tables you want to download',
                        required=True)
    parser.add_argument('-d',
                        '--date_filter',
                        dest='date_filter',
                        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
                        action='store',
                        required=False,
                        default='2018-01-01')
    parser.add_argument('-iv',
                        '--include_visit_type',
                        dest='include_visit_type',
                        action='store_true',
                        help='Specify whether to include visit types for '
                             'generating the training data')
    parser.add_argument('-ip',
                        '--is_new_patient_representation',
                        dest='is_new_patient_representation',
                        action='store_true',
                        help='Specify whether to generate the sequence of '
                             'EHR records using the new patient representation')
    parser.add_argument('-ib',
                        '--is_classic_bert_sequence',
                        dest='is_classic_bert_sequence',
                        action='store_true',
                        help='Specify whether to generate the sequence of '
                             'EHR records using the classic BERT sequence')
    parser.add_argument('-ev',
                        '--exclude_visit_tokens',
                        dest='exclude_visit_tokens',
                        action='store_true',
                        help='Specify whether or not to exclude the VS and VE tokens')
    parser.add_argument('--include_prolonged_length_stay',
                        dest='include_prolonged_stay',
                        action='store_true',
                        help='Specify whether or not to include the data for the second learning '
                             'objective for Med-BERT')
    parser.add_argument('--include_concept_list',
                        dest='include_concept_list',
                        action='store_true')
    ARGS = parser.parse_args()

    main(ARGS.input_folder, ARGS.output_folder, ARGS.domain_table_list, ARGS.date_filter,
         ARGS.include_visit_type, ARGS.is_new_patient_representation, ARGS.exclude_visit_tokens,
         ARGS.is_classic_bert_sequence, ARGS.include_prolonged_stay, ARGS.include_concept_list)
