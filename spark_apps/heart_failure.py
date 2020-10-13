from pyspark.sql import DataFrame

from spark_apps.spark_app_base import NestedCohortBuilderBase
from spark_apps.type_two_diabietes import TypeTwoDiabetesCohortBuilder
from spark_apps.type_two_diabietes import DEPENDENCY_LIST as t2dm_dependency_list
from spark_apps.parameters import create_spark_args

from utils.spark_utils import *

DIURETIC_CONCEPT_ID = 4186999

PACEMAKER_CONCEPT_IDS = [4244395, 4051938, 2107005, 2313863, 4142917, 2106987, 2107003, 4232614,
                         2106988, 2313861, 2313855, 4204395, 42738824, 42742524, 2001980, 35622031,
                         42738822, 2107027, 4215909, 2211371, 4049403, 42738819, 42738820, 2313948,
                         2313860]

HEART_FAILURE_CONCEPTS = [45773075, 45766964, 45766167, 45766166, 45766165, 45766164, 44784442,
                          44784345, 44782733,
                          44782728, 44782719, 44782718, 44782713, 44782655, 44782428, 43530961,
                          43530643, 43530642,
                          43022068, 43022054, 43021842, 43021841, 43021840, 43021826, 43021825,
                          43021736, 43021735,
                          43020657, 43020421, 40486933, 40482857, 40481043, 40481042, 40480603,
                          40480602, 40479576,
                          40479192, 37311948, 37309625, 37110330, 36717359, 36716748, 36716182,
                          36713488, 36712929,
                          36712928, 36712927, 35615055, 4327205, 4311437, 4307356, 4284562, 4273632,
                          4267800, 4264636,
                          4259490, 4242669, 4233424, 4233224, 4229440, 4215802, 4215446, 4206009,
                          4205558, 4199500,
                          4195892, 4195785, 4193236, 4185565, 4177493, 4172864, 4142561, 4141124,
                          4139864, 4138307,
                          4124705, 4111554, 4108245, 4108244, 4103448, 4079695, 4079296, 4071869,
                          4030258, 4023479,
                          4014159, 4009047, 4004279, 3184320, 764877, 764876, 764874, 764873,
                          764872, 764871, 762003,
                          762002, 444101, 444031, 443587, 443580, 442310, 439846, 439698, 439696,
                          439694, 319835,
                          316994, 316139, 314378, 312927]

DOMAIN_TABLE_LIST = ['condition_occurrence', 'drug_exposure', 'procedure_occurrence']
DEPENDENCY_LIST = ['person', 'condition_occurrence', 'visit_occurrence', 'drug_exposure',
                   'procedure_occurrence', 'concept', 'concept_relationship', 'concept_ancestor']

TARGET_COHORT_NAME = 'target_cohort'
DEFAULT_COHORT_NAME = 'heart_failure'
PERSON = 'person'
VISIT_OCCURRENCE = 'visit_occurrence'
CONDITION_OCCURRENCE = 'condition_occurrence'
DIURETICS_CONCEPT = 'diuretics_concepts'

NUM_OF_DIAGNOSIS_CODES = 3


class HeartFailureCohortBuilder(NestedCohortBuilderBase):

    def preprocess_dependencies(self):
        diuretics_concepts = self._build_diuretic_concepts()
        diuretics_concepts.createOrReplaceGlobalTempView('diuretics_concepts')
        self._dependency_dict[DIURETICS_CONCEPT] = diuretics_concepts

        condition_occurrence = self.spark.sql("SELECT * FROM global_temp.condition_occurrence") \
            .where(F.col('condition_concept_id').isin(HEART_FAILURE_CONCEPTS))
        condition_occurrence.createOrReplaceGlobalTempView('hf_condition_occurrence')

        self._target_cohort.createOrReplaceGlobalTempView(TARGET_COHORT_NAME)
        self._dependency_dict[TARGET_COHORT_NAME] = self._target_cohort

    def create_incident_cases(self):
        positive_hf_cases = self.spark.sql("""
            WITH hf_patients AS 
            (
                SELECT
                    c.person_id,
                    c.earliest_visit_start_date,
                    c.earliest_visit_occurrence_id,
                    COUNT(c.visit_occurrence_id) OVER(PARTITION BY c.person_id) AS num_of_diagnosis
                FROM
                (
                    SELECT DISTINCT
                        v.person_id,
                        v.visit_occurrence_id,
                        first(DATE(c.condition_start_date)) OVER (PARTITION BY v.person_id 
                            ORDER BY DATE(c.condition_start_date)) AS earliest_condition_start_date,
                        first(DATE(v.visit_start_date)) OVER (PARTITION BY v.person_id 
                            ORDER BY DATE(v.visit_start_date)) AS earliest_visit_start_date,
                        first(v.visit_occurrence_id) OVER (PARTITION BY v.person_id
                            ORDER BY DATE(v.visit_start_date)) AS earliest_visit_occurrence_id
                    FROM global_temp.visit_occurrence AS v
                    JOIN global_temp.hf_condition_occurrence AS c
                        ON v.visit_occurrence_id = c.visit_occurrence_id
                ) c
                WHERE c.earliest_visit_start_date <= c.earliest_condition_start_date
            )

            SELECT
                tc.*,
                1 AS label
            FROM hf_patients AS hf 
            JOIN global_temp.target_cohort AS tc
                ON hf.person_id = tc.person_id AND DATE_ADD(tc.index_date, 30) <= hf.earliest_visit_start_date
            JOIN global_temp.person AS p
                ON tc.person_id = p.person_id
            WHERE hf.num_of_diagnosis  >= {num_of_diagnosis}
        """.format(num_of_diagnosis=NUM_OF_DIAGNOSIS_CODES))

        return self._exclude_cases_with_prior_pacemakers(
            self._exclude_cases_with_prior_diuretics(positive_hf_cases))

    def create_control_cases(self):
        negative_hf_cases = self.spark.sql("""
            SELECT DISTINCT
                tc.*,
                0 AS label
            FROM global_temp.target_cohort AS tc
            LEFT JOIN 
            (
                SELECT DISTINCT 
                    person_id 
                FROM global_temp.hf_condition_occurrence
            ) p
                ON tc.person_id = p.person_id
            WHERE p.person_id IS NULL
        """)

        return self._exclude_cases_with_prior_pacemakers(
            self._exclude_cases_with_prior_diuretics(negative_hf_cases))

    def create_matching_control_cases(self, incident_cases: DataFrame, control_cases: DataFrame):
        """
        Do not match for control and simply what's in the control cases
        :param incident_cases:
        :param control_cases:
        :return:
        """
        return control_cases

    def _build_diuretic_concepts(self):
        build_ancestry_table_for(self.spark, [DIURETIC_CONCEPT_ID]).createOrReplaceGlobalTempView(
            'ancestry_table')
        diuretics_concepts = self.spark.sql("""
        SELECT DISTINCT
            c.*
        FROM global_temp.ancestry_table AS a
        JOIN global_temp.concept_relationship AS cr
            ON a.descendant_concept_id = cr.concept_id_1 AND cr.relationship_id = 'Maps to'
        JOIN global_temp.concept_ancestor AS ca
            ON cr.concept_id_2 = ca.descendant_concept_id
        JOIN global_temp.concept AS c
            ON ca.ancestor_concept_id = c.concept_id
        WHERE c.concept_class_id = 'Ingredient'
        """)
        return diuretics_concepts

    def _exclude_cases_with_prior_diuretics(self, cohort_group: DataFrame):
        cohort_group.createOrReplaceGlobalTempView('cohort_group')

        cohort_group = self.spark.sql("""
        WITH diuretics_user AS (
            SELECT DISTINCT
                de.person_id,
                FIRST(DATE(drug_exposure_start_date)) AS drug_exposure_start_date
            FROM global_temp.drug_exposure AS de 
            JOIN global_temp.diuretics_concepts AS dc
                ON de.drug_concept_id = dc.concept_id
            GROUP BY de.person_id
        )
        SELECT DISTINCT
            c.*
        FROM global_temp.cohort_group AS c 
        LEFT JOIN diuretics_user AS du
            ON c.person_id = du.person_id AND c.index_date > du.drug_exposure_start_date
        WHERE du.person_id IS NULL
        """)

        self.spark.sql("DROP VIEW global_temp.cohort_group")

        return cohort_group

    def _exclude_cases_with_prior_pacemakers(self, cohort_group: DataFrame):
        cohort_group.createOrReplaceGlobalTempView('cohort_group')

        cohort_group = self.spark.sql("""
        WITH pacemaker_user AS (
            SELECT DISTINCT
                de.person_id,
                FIRST(DATE(de.procedure_date)) AS procedure_date
            FROM global_temp.procedure_occurrence AS de 
            WHERE de.procedure_concept_id IN ({procedure_concept_ids})
            GROUP BY de.person_id
        )
        SELECT DISTINCT
            c.*
        FROM global_temp.cohort_group AS c 
        LEFT JOIN pacemaker_user AS pu
            ON c.person_id = pu.person_id AND c.index_date > pu.procedure_date
        WHERE pu.person_id IS NULL
        """.format(procedure_concept_ids=','.join([str(c) for c in PACEMAKER_CONCEPT_IDS])))

        self.spark.sql("DROP VIEW global_temp.cohort_group")

        return cohort_group


def main(cohort_name, input_folder, output_folder, date_lower_bound, date_upper_bound,
         age_lower_bound, age_upper_bound, observation_window, prediction_window,
         index_date_match_window, is_feature_concept_frequency):
    type_2_diabetes = TypeTwoDiabetesCohortBuilder(f'{DEFAULT_COHORT_NAME}_for_{cohort_name}',
                                                   input_folder,
                                                   output_folder,
                                                   date_lower_bound,
                                                   date_upper_bound,
                                                   age_lower_bound,
                                                   age_upper_bound,
                                                   observation_window,
                                                   prediction_window,
                                                   index_date_match_window,
                                                   DOMAIN_TABLE_LIST,
                                                   t2dm_dependency_list,
                                                   False).build().load_cohort()

    cohort_builder = HeartFailureCohortBuilder(type_2_diabetes,
                                               cohort_name,
                                               input_folder,
                                               output_folder,
                                               date_lower_bound,
                                               date_upper_bound,
                                               age_lower_bound,
                                               age_upper_bound,
                                               observation_window,
                                               prediction_window,
                                               index_date_match_window,
                                               DOMAIN_TABLE_LIST,
                                               DEPENDENCY_LIST,
                                               True,
                                               is_feature_concept_frequency)
    cohort_builder.build()


if __name__ == '__main__':
    spark_args = create_spark_args()

    main(spark_args.cohort_name,
         spark_args.input_folder,
         spark_args.output_folder,
         spark_args.date_lower_bound,
         spark_args.date_upper_bound,
         spark_args.lower_bound,
         spark_args.upper_bound,
         spark_args.observation_window,
         spark_args.prediction_window,
         spark_args.index_date_match_window,
         spark_args.is_feature_concept_frequency)
