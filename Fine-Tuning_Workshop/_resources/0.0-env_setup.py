# Databricks notebook source
catalog = "main"
schema = db = "fine_tuning_workshop"
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()

# spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema}")

spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")

# spark.sql(f"GRANT ALL PRIVILEGES ON SCHEMA {schema} TO `{username}`")

# COMMAND ----------

import warnings
import re
import pandas as pd

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# COMMAND ----------

#Return the current cluster id to use to read the dataset and send it to the fine tuning cluster. See https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html#cluster-id
def get_current_cluster_id():
  import json
  return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']


# COMMAND ----------

from datasets import load_dataset
import pandas as pd

if not spark.catalog.tableExists(f"{catalog}.{schema}.drug_extraction_eval_data"):

    hf_dataset_name = "allenai/drug-combo-extraction"

    # Load datasets
    dataset_test = load_dataset(hf_dataset_name, split="test")
    dataset_train = load_dataset(hf_dataset_name, split="train")

    # Convert to pandas DataFrame and extract entities
    def process_dataset(dataset):
        df = pd.DataFrame(dataset)
        df["human_annotated_entities"] = df["spans"].apply(lambda spans: [span["text"] for span in spans])
        return df[["sentence", "human_annotated_entities"]]

    df_test = process_dataset(dataset_test)
    df_train = process_dataset(dataset_train)

    # Convert to Spark DataFrame and save
    spark.createDataFrame(df_test).write.mode("overwrite").saveAsTable("drug_extraction_eval_data")
    spark.createDataFrame(df_train).write.mode("overwrite").saveAsTable("drug_extraction_train_data")

# COMMAND ----------

from pyspark.sql import functions as F

def clean_baseline_prediction_results(df, column_name):
    """
    This function extracts an array from a string in the format [item1, item2]
    only if the string strictly follows the pattern. If the string contains noise,
    it returns an empty array.

    Parameters:
    df (DataFrame): Input Spark DataFrame
    column_name (str): The name of the column that contains string data

    Returns:
    DataFrame: DataFrame with a new column containing the array
    """
    # Ensure the string strictly matches the format of an array-like string
    # This regex requires the whole string to be like [item1, item2] with no other noise
    df = df.withColumn(
        'extracted_array', 
        F.regexp_extract(F.col(column_name), r'^\[([^\]]+)\]$', 1)
    )

    # Split the extracted string into an array, return an empty array if the pattern is not found
    df = df.withColumn(
        f'{column_name}_cleaned', 
        F.when(F.col('extracted_array') == '', F.array())  # If no match, return empty array
         .otherwise(F.split(F.col('extracted_array'), ',\s*'))  # Else split the matched string into an array
    )

    # Drop the intermediate column
    df = df.drop('extracted_array')

    return df


# COMMAND ----------

from pyspark.sql import functions as F

def extract_array_from_string(df, column_name):
    """
    This function extracts an array from a string in the formats:
    1. [item1, item2] (with commas)
    2. ['item1' 'item2'] (with whitespace instead of commas)
    
    It strictly matches these formats. If the string contains noise or does not
    follow the formats, it returns an empty array and removes any quotes around items.

    Parameters:
    df (DataFrame): Input Spark DataFrame
    column_name (str): The name of the column that contains string data

    Returns:
    DataFrame: DataFrame with the transformed column replacing the original column
    """
    # Extract the array-like pattern from the string, matching strictly
    df = df.withColumn(
        'extracted_array', 
        F.regexp_extract(F.col(column_name), r'^\[([^\]]+)\]$', 1)
    )

    # Check if the string uses commas or whitespace to separate items and split accordingly
    df = df.withColumn(
        column_name,  # Overwriting the original column
        F.when(F.col('extracted_array') == '', F.array())  # If no match, return empty array
         .otherwise(
             F.when(
                 F.col('extracted_array').contains(','),  # If commas, split on ",\s*"
                 F.split(F.col('extracted_array'), ',\s*')
             ).otherwise(
                 F.split(F.col('extracted_array'), '\s+')  # If whitespace, split on whitespace
             )
         )
    )

    # Remove any single quotes around each element in the array
    df = df.withColumn(
        column_name,
        F.expr(f"transform({column_name}, x -> regexp_replace(x, \"'\", ''))")
    )

    # Drop the intermediate column
    df = df.drop('extracted_array')

    return df


# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS main.fine_tuning_workshop.baseline_eval_results

# COMMAND ----------

# # Load Baseline Predictions CSV
# current_directory = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# preran_baseline_result_csv_path = "file:/Workspace/Users/jordan.soldo@databricks.com/Fine-Tuning Workshop 1b/_resources/baseline_eval_results_1b.csv"

# preran_baseline_result_df = spark.read.csv(
#     preran_baseline_result_csv_path,
#     header=True,
#     inferSchema=True,
#     sep="|",
#     multiLine=True
# )

# preran_baseline_result_df = extract_array_from_string(preran_baseline_result_df, 'human_annotated_entities')
# preran_baseline_result_df = extract_array_from_string(preran_baseline_result_df, 'baseline_predictions_cleaned')


# display(preran_baseline_result_df)


# COMMAND ----------

# # Load Baseline Predictions CSV
# current_directory = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()

# preran_baseline_result_csv_path = "file:/Workspace" + current_directory.rsplit('/', 1)[0] + "/_resources/baseline_eval_results_1b.csv"

# preran_baseline_result_df = spark.read.csv(
#     preran_baseline_result_csv_path,
#     header=True,
#     inferSchema=True,
#     sep="|",
#     multiLine=True
# )

# preran_baseline_result_df = extract_array_from_string(preran_baseline_result_df, 'human_annotated_entities')
# preran_baseline_result_df = extract_array_from_string(preran_baseline_result_df, 'baseline_predictions_cleaned')

# preran_baseline_result_df.write.mode("overwrite").saveAsTable("baseline_eval_results")

# COMMAND ----------

from mlflow.tracking import MlflowClient

def get_latest_model_version(model_name):
    mlflow_client = MlflowClient()
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

# llm_model = base_model_name = "llama_v3_2_3b_instruct"
# serving_endpoint_baseline_name = "ft_workshop_baseline_llm"
# latest_version = get_latest_model_version(f"system.ai.{base_model_name}")

# w = WorkspaceClient()
# endpoint_config = EndpointCoreConfigInput(
#     name=serving_endpoint_baseline_name,
#     served_entities=[
#         ServedEntityInput(
#             entity_name=f"system.ai.{llm_model}", #Make sure you're using the same base model as the one you're fine-tuning on for relevant evaluation!
#             entity_version=latest_version,
#             min_provisioned_throughput=0, # The minimum tokens per second that the endpoint can scale down to.
#             max_provisioned_throughput=100,# The maximum tokens per second that the endpoint can scale up to.
#             scale_to_zero_enabled=True
#         )
#     ]
# )

# existing_endpoint = next(
#     (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_baseline_name), None
# )
# if existing_endpoint == None:
#     print(f"Creating the endpoint {serving_endpoint_baseline_name}, this will take a few minutes to package and deploy the LLM...")
#     w.serving_endpoints.create_and_wait(name=serving_endpoint_baseline_name, config=endpoint_config)
# else:
#   print(f"endpoint {serving_endpoint_baseline_name} already exist")

# COMMAND ----------

from sklearn.metrics import precision_score, recall_score
import pandas as pd


def _compute_precision_recall(prediction, ground_truth):
    if prediction is None:
        prediction = []
    if ground_truth is None:
        ground_truth = []
    prediction_set = set([str(drug).lower() for drug in prediction])
    ground_truth_set = set([str(drug).lower() for drug in ground_truth])
    all_elements = prediction_set.union(ground_truth_set)

    # Convert sets to binary lists
    prediction_binary = [int(element in prediction_set) for element in all_elements]
    ground_truth_binary = [int(element in ground_truth_set) for element in all_elements]
    
    precision = precision_score(ground_truth_binary, prediction_binary)
    recall = recall_score(ground_truth_binary, prediction_binary)

    return precision, recall

# def _compute_precision_recall(prediction, ground_truth):
#     prediction_set = set([str(drug).lower() for drug in prediction])
#     ground_truth_set = set([str(drug).lower() for drug in ground_truth])
#     all_elements = prediction_set.union(ground_truth_set)

    # Convert sets to binary lists
    # prediction_binary = [int(element in prediction_set) for element in all_elements]
    # ground_truth_binary = [int(element in ground_truth_set) for element in all_elements]
    
    # precision = precision_score(ground_truth_binary, prediction_binary)
    # recall = recall_score(ground_truth_binary, prediction_binary)

    # return precision, recall

def _precision_recall_series(row, prediction_col, ground_truth_col):
    precision, recall = _compute_precision_recall(row[prediction_col], row[ground_truth_col])
    return pd.Series([precision, recall], index=['precision', 'recall'])

def calculate_metrics(df, prediction_col, ground_truth_col):
    df = df.toPandas()
    df[['precision', 'recall']] = df.apply(_precision_recall_series, args=(prediction_col, ground_truth_col), axis=1)
    return df['precision'].mean(), df['recall'].mean()

# COMMAND ----------

def clean_finetuned_results(finetuned_prediction_df):
    from pyspark.sql.functions import udf
    from pyspark.sql.types import ArrayType, StringType
    import ast


    def string_to_array(s):
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []

    convert_string_to_array_udf = udf(string_to_array, ArrayType(StringType()))

    finetuned_prediction_cleaned_df = finetuned_prediction_df.withColumn('finetuned_predictions_cleaned', convert_string_to_array_udf('finetuned_predictions')).drop('finetuned_predictions')

    return finetuned_prediction_cleaned_df

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

import json
import re
# Extract the json array from the text, removing potential noise
def extract_json_array(text):
    # Use regex to find a JSON array within the text
    match = re.search(r'(\[.*?\])', text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return []


# Define the UDF
clean_llm_output = udf(extract_json_array, ArrayType(StringType()))

# COMMAND ----------

# DBTITLE 1,reset experiment
import mlflow

def reset_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

    # Get the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # List all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    # Delete each run
    for run_id in runs['run_id']:
        mlflow.delete_run(run_id)

    # print(f"All runs in experiment '{experiment_name}' have been deleted.")

# COMMAND ----------

#Helper fuinction to Wait for the fine tuning run to finish
def wait_for_run_to_finish(run):
  import time
  for i in range(300):
    events = run.get_events()
    for e in events:
      if "FAILED" in e.type or "EXCEPTION" in e.type:
        raise Exception(f'Error with the fine tuning run, check the details in run.get_events(): {e}')
    if events[-1].type == 'COMPLETED':
      print('Run finished')
      display(events)
      return events
    if i % 30 == 0:
      print(f'waiting for run {run.name} to complete...')
    time.sleep(10)


#Format answer, converting MD to html
def display_answer(answer):
  import markdown
  displayHTML(markdown.markdown(answer['choices'][0]['message']['content']))

# COMMAND ----------

# Helper function
def get_latest_model_version(model_name):
    from mlflow.tracking import MlflowClient
    mlflow_client = MlflowClient(registry_uri="databricks-uc")
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version

# COMMAND ----------

import matplotlib.pyplot as plt

def plot_comparison(experiment_id):
  # Corrected code to search runs using experiment_id
  runs = mlflow.search_runs([experiment_id])

  # Extract precision and recall metrics
  metrics_df = runs[["metrics.precision", "metrics.recall"]]

  # Plot precision and recall as a bar graph
  plt.figure(figsize=(10, 5))
  bar_width = 0.35
  index = range(len(metrics_df))
  bars1 = plt.bar(index, metrics_df["metrics.precision"], bar_width, label="Precision")
  bars2 = plt.bar([i + bar_width for i in index], metrics_df["metrics.recall"], bar_width, label="Recall")
  plt.xlabel("Run Index")
  plt.ylabel("Metric Value")
  plt.title("Precision and Recall Metrics Comparison")
  plt.xticks([i + bar_width / 2 for i in index], ["baseline", "finetuned"])
  plt.legend()

  # Add value labels on top of the bars
  for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom') 

  for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom') 

  plt.show()

# COMMAND ----------

insert_statement = """
INSERT INTO main.fine_tuning_workshop.baseline_eval_results (sentence, human_annotated_entities, baseline_predictions_cleaned) VALUES
('A randomized Phase III trial demonstrated noninferiority of APF530 500 mg SC ( granisetron 10 mg ) to intravenous palonosetron 0.25 mg in preventing CINV in patients receiving MEC or HEC in acute ( 0 - 24 hours ) and delayed ( 24 - 120 hours ) settings , with activity over 120 hours .', ARRAY("granisetron", "palonosetron"), ARRAY("APF530", "500", "mg", "SC", "granisetron", "10", "mg", "MEC", "HEC")),
('Classical NSAIDs are still the most logical choice for agents that will slow the progression or delay the onset of AD and other neurodegenerative diseases despite failures of naproxen , celecoxib and rofecoxib in AD clinical trials .', ARRAY("naproxen", "celecoxib", "rofecoxib"), ARRAY("Alvocarboxic", "acid", "diclofenac", "ibuprofen", "naproxen", "celecoxib", "rofecoxib")),
('Between 1987 and 2003 , patients 18 years old were given adjuvant chemotherapy consisting of one of two '' paediatric '' regimens ( depending on the time of presentation ) and craniospinal local-boost radiotherapy : regimen A ( n = 12 ) , vincristine ( VCR ) , intrathecal and/or intravenous methotrexate and conventional radiotherapy ; or regimen B ( n = 11 ) sequencing intensive doses of multiple agents followed by hyperfractionated accelerated radiotherapy ( HART ) .', ARRAY("vincristine", "methotrexate"), ARRAY()),
('The US Food and Drug Administration ( FDA ) recommends that " concomitant use of drugs that inhibit CYP2C19 ( e.g. , omeprazole ) should be discouraged . " As the presence of PPIs and clopidogrel in plasma is short lived , separation by 12 - 20 h should in theory prevent competitive inhibition of CYP metabolism and minimize any potential , though unproven , clinical interaction .', ARRAY("omeprazole", "clopidogrel"), ARRAY("Omeprazole", "Clopidogrel")),
('Long-term overall- and progression-free survival after pentostatin , cyclophosphamide and rituximab therapy for indolent non-Hodgkin lymphoma .', ARRAY("cyclophosphamide", "rituximab", "pentostatin"), ARRAY("Long-term", "overall-", "and", "progression-free", "survival", "after", "pentostatin", "cyclophosphamide", "and", "rituximab", "therapy", "for", "indolent", "non-Hodgkin", "lymphoma")),
('Once-weekly isoniazid and rifapentine for 3 months is a treatment option in persons with human immunodeficiency virus and latent tuberculosis infection .', ARRAY("isoniazid", "rifapentine"), ARRAY("Isoniazid", "Rifampicin", "Isoniazid", "Rifapentine")),
('We investigated the effects of these inhibitors on other anticancer drugs including docetaxel , vinblastine , doxorubicin , 10-Hydroxycamptothecin ( 10-HCPT ) and cisplatin and find that both inhibitors induces DU145-TxR cells to be more sensitive only to the microtubule-targeting drugs ( paclitaxel , docetaxel and vinblastine ) .', ARRAY("docetaxel", "vinblastine", "doxorubicin", "cisplatin", "paclitaxel", "docetaxel", "vinblastine"), ARRAY("docetaxel", "vinblastine", "doxorubicin", "10-Hydroxycamptothecin", "(10-HCPT)", "cisplatin", "paclitaxel", "docetaxel", "vinblastine")),
('Group B : From January , 1976 to December , 1980 , 55 evaluable patients participated in a consecutive study that added Adriamycin ( doxorubicin ) and cyclophosphamide to the former induction regimen .', ARRAY("doxorubicin", "cyclophosphamide"), ARRAY("Adriamycin", "Cyclophosphamide")),
('When given in repeated doses from 6 h on , 50 g followed by 12.5 g at 6-h intervals , charcoal shortened the serum half-life of amitriptyline by 20 % and that of nortriptyline by 35 % ( p less than 0.05 ) .', ARRAY("amitriptyline", "nortriptyline"), ARRAY()),
('The median overall survival time was 22.7 months in the docetaxel arm and 22.4 months in the paclitaxel arm .', ARRAY("docetaxel", "paclitaxel"), ARRAY("Docetaxel", "Paclitaxel")),
('We hypothesized that Aurora A kinase ( AK ) contributes to castrate resistance in prostate cancer ( PCa ) and that inhibiting AK with alisertib can resensitize PCa cells to androgen receptor ( AR ) inhibitor abiraterone .', ARRAY("alisertib", "abiraterone"), ARRAY("Alisertib", "Aurora", "A", "kinase", "inhibitor", "Androgen", "receptor", "inhibitor", "Abiraterone")),
('Tanespimycin plus trastuzumab is well tolerated and has antitumor activity in patients with HER-2 + breast cancer whose tumors have progressed during treatment with trastuzumab .', ARRAY("Tanespimycin", "trastuzumab", "trastuzumab"), ARRAY("Trastuzumab", "Tanespimycin")),
('It contains data on prophylaxis with mefloquine ( n = 48,264 ) , with chloroquine ( 6,752 ) , with chloroquine plus proguanil ( 19,727 ) , and with no prophylaxis ( 3,871 ) .', ARRAY("mefloquine", "chloroquine", "chloroquine", "proguanil"), ARRAY("Mefloquine", "Chloroquine", "Chloroquine", "plus", "proguanil", "No", "prophylaxis")),
('We hypothesized that the brain damage mitigating effect of mild hypothermia after cardiac arrest can be enhanced with thiopental loading , and even more so with the further addition of phenytoin and methylprednisolone .', ARRAY("phenytoin", "methylprednisolone"), ARRAY("Thiopental", "Methylprednisolone")),
('Caspase-independent mechanisms , mainly based on increased oxidative stress , result from 2-methoxyestradiol , Artesunate , ascorbic acid , Dihydroartemisinin , Evodiamine , b-AP15 , VLX1570 , Erw-ASNase , and TAK-242 .', ARRAY("Artesunate", "Dihydroartemisinin"), ARRAY("Artesunate", "Ascorbic", "acid", "Dihydroartemisinin", "Evodiamine", "b-AP15", "TAK-242", "Caspase-independent", "mechanisms", "2-methoxyestradiol", "VLX1570")),
('In the xenograft model , more augmented effects were achieved when bortezomib was combined with gemcitabine than gemcitabine alone .', ARRAY("bortezomib", "gemcitabine", "gemcitabine"), ARRAY("Abraxane", "Bortezomib", "Gemcitabine")),
('The effectiveness of combination therapy with afatinib and bevacizumab may provide a new therapeutic option for these patients .', ARRAY("afatinib", "bevacizumab"), ARRAY("Bevacizumab", "Afatinib")),
('Clinical activity of enzalutamide versus docetaxel in men with castration-resistant prostate cancer progressing after abiraterone .', ARRAY("enzalutamide", "docetaxel", "abiraterone"), ARRAY("Enzalutamide", "Docetaxel")),
('The use of cyclophosphamide in patients with NSVN is controversial , but recent retrospective data suggest that those treated with prednisone and cyclophosphamide from the outset fare better than those initially treated only with prednisone .', ARRAY("cyclophosphamide", "prednisone", "cyclophosphamide", "prednisone"), ARRAY("Cyclophosphamide", "Prednisone")),
('This was a multi-center randomized , two-armed , double-blinded phase II study comparing cediranib plus gefitinib versus cediranib plus placebo in subjects with first relapse/first progression of glioblastoma following surgery and chemoradiotherapy .', ARRAY("cediranib", "gefitinib", "cediranib"), ARRAY()),
('This study aims to compare the biological , molecular , pharmacological , and clinical characteristics of these three treatment modalities for SARS-COV-2 infections , Chloroquine and Hydroxychloroquine , Convalescent Plasma , and Remdesivir .', ARRAY("Chloroquine", "Hydroxychloroquine", "Remdesivir"), ARRAY("Chloroquine", "Hydroxychloroquine", "Remdesivir", "Convalescent", "Plasma")),
('Patients received a mean ( ±standard deviation ) of 8.8 ± 4.9 intravitreal bevacizumab injections prior to the switch to intravitreal ranibizumab .', ARRAY("bevacizumab", "ranibizumab"), ARRAY("Bevacizumab", "Ranibizumab")),
('Combinations of penicillin and streptomycin and penicillin and amikacin were synergistic only against those strains that were not highly resistant to streptomycin and kanamycin , respectively .', ARRAY("streptomycin", "amikacin", "streptomycin", "kanamycin", "penicillin", "penicillin"), ARRAY()),
('Treatment with a combination chemotherapeutic regimen consisting of cyclophosphamide , vincristine , and dacarbazine for malignant paraganglioma with hepatic metastasis is reported .', ARRAY("cyclophosphamide", "vincristine", "dacarbazine"), ARRAY("Cyclophosphamide", "Vincristine", "Dacarbazine")),
('The aim of this study was to evaluate the in vitro effect of single antibiotic ( ciprofloxacin , ceftazidime , or ampicillin ) treatment on adherence of Escherichia coli and Enterococcus to plastic stents .', ARRAY("ciprofloxacin", "ceftazidime", "ampicillin"), ARRAY("ciprofloxacin", "ceftazidime", "ampicillin")),
('Paclitaxel and docetaxel as single agents have yielded overall response rates of 7 % to 56 % , depending on whether the patients have received prior chemotherapy for metastatic disease .', ARRAY("Paclitaxel", "docetaxel"), ARRAY("Paclitaxel", "Docetaxel")),
('The most effective treatment regimens for advanced nonseminomatous testicular tumors employ vinblastine , CDDP and bleomycin and adjunctive surgery .', ARRAY("vinblastine", "bleomycin"), ARRAY("vinblastine", "CDDP", "bleomycin")),
('27 patients suffering from disseminated carcinoma of the breast with at least two visceral metastases , and two had become resistant to conventional chemotherpy and hormones , received a combination of , in the present trial , vincristine followed by cyclophosphamide with 5-fluoro-uracil .', ARRAY("vincristine", "cyclophosphamide", "5-fluoro-uracil"), ARRAY("Vincristine", "Cyclophosphamide", "5-fluoro-uracil")),
('We previously demonstrated that the combination of oral estramustine ( 15 mg/kg/day ) and oral etoposide ( 50 mg/m2/day ) is effective first-line therapy for the treatment of hormone refractory prostate cancer .', ARRAY("estramustine", "etoposide"), ARRAY("Oral", "estramustine", "Oral", "etoposide")),
('Therapy consisted of bendamustine ( 70 mg/m(2 ) ) for 2 consecutive days every 28 days , and ofatumumab 300 mg on day 1 and 1000 mg on day 8 during the first cycle , and 1000 mg on day 1 subsequently .', ARRAY("bendamustine", "ofatumumab"), ARRAY("bendamustine", "therapy", "ofatumumab")),
('All consecutive patients with advanced BTC received the GEMOX regimen in a setting outside a study : gemcitabine 1,000 mg/m(2 ) on day 1 , and oxaliplatin 100 mg/m(2 ) on day 2 , treatment repeated every 2 weeks until progression or unacceptable toxicity .', ARRAY("gemcitabine", "oxaliplatin"), ARRAY("Gemcitabine", "Oxaliplatin")),
('Recent randomized controlled trials ( RCT ) have failed to demonstrate the efficacy of widely used therapies , such as rituximab plus intravenous immunoglobulin or proteasome inhibition ( bortezomib ) , reinforcing a great need for new therapeutic concepts .', ARRAY("rituximab", "intravenous", "immunoglobulin", "bortezomib"), ARRAY()),
('The aim of this study was to find an experimental model of a donor-recipient rat strain combination that , under triple drug immunosuppressive treatment ( methylprednisolone , cyclosporine , and azathioprine ) , would develop chronic rejection within a few weeks .', ARRAY("methylprednisolone", "cyclosporine", "azathioprine"), ARRAY("Azathioprine", "Cyclosporine", "Methylprednisolone")),
('In FOLL05 trial , R-CHOP was compared with R-CVP ( cyclophosphamide , vincristine , prednisone ) and R-FM ( fludarabine , mitoxantrone ) .', ARRAY("R-CHOP", "cyclophosphamide", "vincristine", "prednisone", "fludarabine", "mitoxantrone"), ARRAY("R-CHOP", "R-CVP", "R-FM")),
('All patients underwent 3 cycles of neoadjuvant gemcitabine , paclitaxel , and capecitabine .', ARRAY("gemcitabine", "paclitaxel", "capecitabine"), ARRAY("Gemcitabine", "Paclitaxel", "Capecitabine")),
('The aim of the present investigation was to study and characterize the effect of voriconazole on the fungicidal activity of amphotericin B.', ARRAY("voriconazole", "amphotericin", "B."), ARRAY("Amphotericin", "B", "Voriconazole")),
('Dual Therapy with Aspirin and Cilostazol May Improve Platelet Aggregation in Noncardioembolic Stroke Patients : A Pilot Study .', ARRAY("Aspirin", "Cilostazol"), ARRAY("Aspirin", "Cilostazol")),
('Gemcitabine and vinorelbine have shown activity in the first-line setting .', ARRAY("Gemcitabine", "vinorelbine"), ARRAY("Gemcitabine", "Vinorelbine")),
('The combination of tenofovir disoproxil fumarate ( TDF ) plus emtricitabine ( FTC ) is used extensively to treat HIV infection and also has potent activity against hepatitis B virus ( HBV ) infection .', ARRAY("tenofovir", "disoproxil", "fumarate", "emtricitabine"), ARRAY("tenofovir", "disoproxil", "fumarate", "emtricitabine")),
('HRQOL was better in Japanese postmenopausal women treated with tamoxifen than those treated with exemestane or anastrozole .', ARRAY("tamoxifen", "exemestane", "anastrozole"), ARRAY("Tamoxifen", "Exemestane", "Anastrozole")),
('This was driven by the relative advantage of weight loss compared with rosiglitazone , glimepiride , and insulin glargine , and administration frequency compared with exenatide .', ARRAY("rosiglitazone", "glimepiride"), ARRAY("Rosiglitazone", "Glimepiride", "Exenatide", "Metformin")),
('Pegylated liposomal doxorubicin plus cyclophosphamide followed by paclitaxel as primary chemotherapy in elderly or cardiotoxicity-prone patients with high-risk breast cancer : results of the phase II CAPRICE study .', ARRAY("doxorubicin", "cyclophosphamide", "paclitaxel"), ARRAY("doxorubicin", "cyclophosphamide", "paclitaxel")),
('We randomly assigned 410 women with advanced ovarian cancer and residual masses larger than 1 cm after initial surgery to receive cisplatin ( 75 mg per square meter of body-surface area ) with either cyclophosphamide ( 750 mg per square meter ) or paclitaxel ( 135 mg per square meter over 24 hours ) .', ARRAY("cisplatin", "cyclophosphamide", "paclitaxel"), ARRAY("Paclitaxel", "Cyclophosphamide", "Cisplatin")),
('Changes in cholesterol absorption and cholesterol synthesis caused by ezetimibe and/or simvastatin in men .', ARRAY("ezetimibe", "simvastatin"), ARRAY("ezetimibe", "simvastatin")),
('The US government regulated precursor chemicals , ephedrine and pseudoephedrine , multiple times to limit methamphetamine production/availability and thus methamphetamine problems .', ARRAY("pseudoephedrine", "methamphetamine", "methamphetamine"), ARRAY()),
('We now present the long-term follow-up findings of a randomized phase III study on the addition of six cycles of procarbazine , lomustine , and vincristine ( PCV ) chemotherapy to radiotherapy ( RT ) .', ARRAY("procarbazine", "lomustine", "vincristine", "radiotherapy"), ARRAY()),
('A phase II clinical trial was conducted to evaluate the efficacy and toxicity of vinorelbine plus gemcitabine in very old patients with inoperable ( stage IIIb or IV ) NSCLC .', ARRAY("vinorelbine", "gemcitabine"), ARRAY("Vinorelbine", "Gemcitabine")),
('Tobramycin was probably less effective than gentamicin in combination with the penicillinase-resistant penicillins against enterococci .', ARRAY("Tobramycin", "gentamicin", "penicillins"), ARRAY("tobramycin", "gentamicin", "penicillins", "penicillase-resistant", "penicillins", "enterococci")),
('Despite the absence of strong data , entecavir and telbivudine seem to be the preferred options for nucleoside analogue-naive CHB patients with chronic kidney disease , depending on viraemia and severity of renal dysfunction .', ARRAY("entecavir", "telbivudine"), ARRAY("Entecavir", "Telbivudine")),
('Results from clinical trials evaluating agents such as ixabepilone , albumin-bound paclitaxel , capecitabine , vinorelbine , pemetrexed , and irinotecan are presented .', ARRAY("ixabepilone", "paclitaxel", "capecitabine", "vinorelbine", "pemetrexed", "irinotecan"), ARRAY("albumin-bound", "paclitaxel", "capecitabine", "vinorelbine", "pemetrexed", "ixabepilone", "irinotecan")),
('The clinical studies provide evidence that combined fluticasone/formoterol is more efficacious than fluticasone or formoterol given alone , and provides similar improvements in lung function to fluticasone ( Flixotide ( ® ) ) and formoterol ( Foradil ( ® ) ) administered concurrently .', ARRAY("fluticasone", "formoterol", "fluticasone", "formoterol"), ARRAY()),
('Prediction of risk of distant recurrence using the 21-gene recurrence score in node-negative and node-positive postmenopausal patients with breast cancer treated with anastrozole or tamoxifen : a TransATAC study .', ARRAY("anastrozole", "tamoxifen"), ARRAY("Anastrozole", "Tamoxifen")),
('When novel DMARDs were used as monotherapies , greater ACR20/50/70 responses were observed with tocilizumab than with anti-tumor necrosis factor agents ( aTNF ) or tofacitinib .', ARRAY("tocilizumab", "tofacitinib"), ARRAY()),
('Paclitaxel showed synergistic anti-proliferative impacts with fulvestrant .', ARRAY("Paclitaxel", "fulvestrant"), ARRAY("Paclitaxel", "Fulvestrant")),
('The experimental protocol was as follows : ( 1 ) at time zero , intravenous infusion of ritodrine was begun ; ( 2 ) at 120 min , 2 % lidocaine was given epidurally to achieve a sensory level of at least T6 ; and ( 3 ) at 135 min , an intravenous bolus of either ephedrine , phenylephrine , or normal saline-control was given , followed by a continuous intravenous infusion of the same agent for 30 min .', ARRAY("ritodrine", "lidocaine", "phenylephrine", "ephedrine"), ARRAY("ritodrine", "lidocaine", "ephedrine", "phenylephrine")),
('compared to busulfan and cyclophosphamide as conditioning regimen for allogeneic stem cell transplant from matched siblings and unrelated donors for acute myeloid leukemia .', ARRAY("busulfan", "cyclophosphamide"), ARRAY("Busulfan", "Cyclophosphamide")),
('Posttransplant immunosuppression to prevent graft-versus-host disease ( GVHD ) consisted of cyclosporine and prednisone ( CSA/PSE ) .', ARRAY("cyclosporine", "prednisone"), ARRAY("cyclosporine", "prednisone")),
('Tadalafil 20 mg was preferred to sildenafil 50 mg for the initiation of ED therapy in this study population .', ARRAY("Tadalafil", "sildenafil"), ARRAY("tadalafil", "sildenafil")),
('Reactivity of factor IXa with basic pancreatic trypsin inhibitor is enhanced by low molecular weight heparin ( enoxaparin ) .', ARRAY("heparin", "enoxaparin"), ARRAY("Warfarin", "Enoxaparin")),
('Either 5-fluorouracil ( 5-FU ) and leucovorin for 6 months or 5-FU and levamisole for 12 months are currently considered standard adjuvant treatment for stage III colorectal cancer .', ARRAY("leucovorin", "levamisole", "5-FU", "5-FU"), ARRAY("5-FU", "5-fluorouracil", "levamisole", "levamisole")),
('Postmenopausal women with stage I-IIIA hormone receptor-positive breast cancer , who were disease-free after about 5 years of treatment with an aromatase inhibitor or tamoxifen followed by an aromatase inhibitor , were randomly assigned ( 1:1 ) to receive 5 years of letrozole ( 2·5 mg orally per day ) or placebo .', ARRAY("tamoxifen", "letrozole"), ARRAY("Artemether", "Letrozole", "Tamoxifen")),
('Efficacy of sorafenib combined with the pan-CDK inhibitor flavopiridol was tested both in vitro and in xenograft experiments .', ARRAY("sorafenib", "flavopiridol"), ARRAY("Flavopiridol", "Sorafenib")),
('Codelivery of sorafenib and curcumin by directed self-assembled nanoparticles enhances therapeutic effect on hepatocellular carcinoma .', ARRAY("sorafenib", "curcumin"), ARRAY("Curcumin", "Sorafenib")),
('Single treatment of closantel plus albendazole mixture reduced egg counts in camels by 100 % , 100 % , 98 % and 77 % for Haemonchus longistipes , Ascaris spp . , Monezia expansa and Fasciola hepatica , respectively .', ARRAY("closantel", "albendazole"), ARRAY()),
('In contrast to patients receiving lidocaine , older patients receiving tetracaine experienced significantly less overall pain and discomfort , unpleasant taste , and dyspnea .', ARRAY("lidocaine", "tetracaine"), ARRAY("Hydrocortisone", "Lidocaine", "Tetracaine")),
('Second-line chemotherapy was also ineffective ; therefore , the bevacizumab and irinotecan were given after a third gross-total resection of the tumor .', ARRAY("bevacizumab", "irinotecan"), ARRAY("Bevacizumab", "Irinotecan")),
('In unselected patients with advanced HCC immunotherapeutics , namely the programmed cell death-1 ( PD-1 ) antibodies , nivolumab and pembrolizumab have shown promising efficacy in therapy-naïve , as well as pre-treated patients with advanced HCC .', ARRAY("nivolumab", "pembrolizumab"), ARRAY("Immunotherapy", "Nivolumab", "Pembrolizumab")),
('Treatment with rapamycin and paclitaxel resulted in decreased phosphorylation of S6 and 4E-BP1 , two critical downstream targets of the mTOR pathway .', ARRAY("rapamycin", "paclitaxel"), ARRAY("rapamycin", "paclitaxel")),
('Conversely , the IMbrave150 trial recently showed that , among patients with previously untreated unresectable HCC , treatment with atezolizumab plus bevacizumab resulted in significantly longer overall survival and progression-free survival compared to sorafenib monotherapy .', ARRAY("atezolizumab", "bevacizumab", "sorafenib"), ARRAY("Atezolizumab", "Bevacizumab", "Sorafenib")),
('Evaluation of tamoxifen plus letrozole with assessment of pharmacokinetic interaction in postmenopausal women with metastatic breast cancer .', ARRAY("tamoxifen", "letrozole"), ARRAY("Tamoxifen", "Letrozole")),
('Moreover , the BCR/ABL1 inhibitors nilotinib and ponatinib were found to decrease STAT5 activity and CD25 expression in KU812 cells and primary CML LSCs .', ARRAY("nilotinib", "ponatinib"), ARRAY("Nilotinib", "Ponatinib")),
('The EVERLAR study reports prospective data of somatostatin analogue in combination with everolimus in nonfunctioning gastrointestinal neuroendocrine tumors suggesting meaningful activity and favorable toxicity profile that supports drug combination in this setting .', ARRAY("somatostatin", "everolimus"), ARRAY("Somatostatin", "analogue", "Everolimus")),
('Synergism between anti-HER2 monoclonal antibody ( trastuzumab ) and paclitaxel has been shown in vitro and in vivo .', ARRAY("trastuzumab", "paclitaxel"), ARRAY("Trastuzumab", "Paclitaxel")),
('Between September 1998 and December 1999 , we treated 25 patients at risk for post-operative renal dysfunction ( high-risk basiliximab group ) with the new induction regimen and another 33 patients not at risk ( low-risk CsA group ) for renal dysfunction with our standard cyclosporine protocol .', ARRAY("basiliximab", "cyclosporine"), ARRAY("Abatacept", "Basiliximab", "Cyclosporine")),
('Cardiac events occurred after posaconazole administration , incriminating posaconazole use , alone or in combination with voriconazole , as the culpable agent .', ARRAY("posaconazole", "posaconazole", "voriconazole"), ARRAY("Posaconazole", "Voriconazole")),
('Prognostic value of expression of Kit67 , p53 , TopoIIa and GSTP1 for curatively resected advanced gastric cancer patients receiving adjuvant paclitaxel plus capecitabine chemotherapy .', ARRAY("paclitaxel", "capecitabine"), ARRAY("Paclitaxel", "Topoisomerase", "IIa", "GSTP1", "Prognostic", "value", "Curative", "resection", "Gastric", "cancer", "Adjuvant", "chemotherapy")),
('The standard adjuvant treatment of colon cancer is fluorouracil plus leucovorin ( FL ) .', ARRAY("fluorouracil", "leucovorin"), ARRAY()),
('Effects of ultrasound were similar at baseline and after propranolol but increased after phenylephrine .', ARRAY("propranolol", "phenylephrine"), ARRAY("Propranolol", "Phenylephrine")),
('Thirty-four patients ( 4 patients with stage IIIb , 30 patients in stage IV ) , with median age 66 and performance status 0 - 1 , were administered paclitaxel , 175 mg/m(2 ) in a 3-h infusion rate on day 1 and vinorelbine , 25 mg/m(2 ) in a 10-min infusion rate on days 1 , and 8 with G-CSF and EPO support .', ARRAY("paclitaxel", "vinorelbine"), ARRAY("Paclitaxel", "Vinorelbine", "G-CSF", "EPO")),
('Phase I study of decitabine alone or in combination with valproic acid in acute myeloid leukemia .', ARRAY("decitabine", "valproic"), ARRAY("Decitabine", "Valproic", "acid")),
('Early discontinuation of clopidogrel results in a transient rebound increase in risk of recurrence in acute coronary syndromes , but there are no published data on any similar rebound effect in patients with TIA or stroke that might inform the design of clinical trials of aspirin and clopidogrel in the acute phase .', ARRAY("clopidogrel", "aspirin", "clopidogrel"), ARRAY("Aspirin", "Clopidogrel")),
('Patients received docetaxel 35 mg/m(2 ) and irinotecan 60 mg/m(2 ) , intravenously , on Days 1 and 8 , every 21 days , until disease progression .', ARRAY("docetaxel", "irinotecan"), ARRAY("Docetaxel", "Irinotecan")),
('Most patients with advanced ovarian cancer achieve a clinical complete remission following cytoreductive surgery and chemotherapy with paclitaxel plus carboplatin .', ARRAY("paclitaxel", "carboplatin"), ARRAY("Paclitaxel", "Carboplatin")),
('Patients were assigned to receive either oxaliplatin ( 100 mg/m(2 ) ) in the SOX arm or docetaxel ( 40 mg/m(2 ) ) in the mDS arm on day 1 of every 3-week cycle .', ARRAY("oxaliplatin", "docetaxel"), ARRAY("Oxaliplatin", "Docetaxel")),
('Agents in clinical trials for subsets of MDS include luspatercept , antibodies targeting CD33 , isocitrate dehydrogenase inhibitors , deacetylase inhibitors , venetoclax , and immunotherapies designed to overcome immune checkpoint inhibition .', ARRAY("luspatercept", "venetoclax"), ARRAY()),
('Possible differences between enalapril and captopril .', ARRAY("enalapril", "captopril"), ARRAY()),
('Exposure of the saphenous veins to imipenem or imipenem combined with amphotericin B had no adverse effects on the viability of the endothelial cells with 12 h exposure .', ARRAY("imipenem", "imipenem", "amphotericin"), ARRAY("Imipenem", "Imipenem-cilastatin", "Imipenem-cilastatin", "disodium", "salt", "Imipenem-cilastatin", "sodium", "Imipenem-cilastatin", "sodium", "citrate")),
('First-line chemotherapy was either R-CHOP ( rituximab , cyclophosphamide , doxorubicin , vincristine and prednisolone ) or CHOP-like-based regimen .', ARRAY("rituximab", "cyclophosphamide", "doxorubicin", "vincristine", "prednisolone"), ARRAY("R-CHOP", "cyclophosphamide", "doxorubicin", "vincristine", "prednisolone")),
('The combination of rituximab with vincristine and 5-day cyclophosphamide is able to produce CR in patients with advanced follicular lymphoma , even in patients resistant to third-generation regimens .', ARRAY("rituximab", "vincristine", "cyclophosphamide"), ARRAY("Rituximab", "Vincristine", "Cyclophosphamide")),
('The results showed that lamotrigine did not produce any change in cognitive function , while carbamazepine produced cognitive dysfunction .', ARRAY("lamotrigine", "carbamazepine"), ARRAY("Carbamazepine", "Lamotrigine")),
('These results confirm that sequential decitabine and carboplatin requires further investigation as a combination treatment for melanoma .', ARRAY("decitabine", "carboplatin"), ARRAY()),
('[ Rhabdomyosarcoma of the urinary bladder : complete remission induced by vinblastine , cis-platinum , and bleomycin ] .', ARRAY("vinblastine", "bleomycin", "cis-platinum"), ARRAY("Rhabdomyosarcoma", "of", "the", "urinary", "bladder", ":", "complete", "remission", "induced", "by", "vinblastine", "cis-platinum", "and", "bleomycin")),
('Recent comparative studies suggest that atenolol ( 200 mg daily ) , metoprolol ( 200 mg daily ) ; acebutolol ( 400 mg daily ) , oxprenolol ( 160 mg daily ) , nadolol ( 80 mg daily ) and timolol ( 20 mg daily ) produce a beneficial clinical response equal to that seen with propranolol ( 160 mg daily ) .', ARRAY("atenolol", "metoprolol", "acebutolol", "oxprenolol", "nadolol", "timolol", "propranolol"), ARRAY("atenolol", "metoprolol", "acebutolol", "oxprenolol", "nadolol", "timolol")),
('All the animals in the placebo group had tumors in each lobe compared with only 43 % each in the dorsolateral ( DLP ) and anterior prostate ( AP ) of the animals treated with raloxifene ( 10 mg/kg/day ) plus nimesulide .', ARRAY("raloxifene", "nimesulide"), ARRAY("Raloxifene", "Nimesulide")),
('Overdose of dolutegravir in combination with tenofovir disaproxil fumarate/emtricitabine in suicide attempt in a 21-year old patient .', ARRAY("dolutegravir", "tenofovir"), ARRAY()),
('Abiraterone acetate and prednisone in the pre- and post-docetaxel setting for metastatic castration-resistant prostate cancer : a mono-institutional experience focused on cardiovascular events and their impact on clinical outcomes .', ARRAY("Abiraterone", "prednisone"), ARRAY("Abiraterone", "acetate", "Prednisone")),
('Nivolumab plus ipilimumab combined therapy is among the most effective therapies for advanced melanoma .', ARRAY("Nivolumab", "ipilimumab"), ARRAY("Ipilimumab", "Nivolumab")),
('Cells that were sequentially exposed to rapamycin and topotecan had significantly higher levels of cleaved caspase-8 , -3 , and PARP compared to those treated with topotecan alone .', ARRAY("rapamycin", "topotecan", "topotecan"), ARRAY("rapamycin", "topotecan", "caspase-8", "caspase-3", "PARP")),
('No cross-resistance was found with conventional drugs , being PNU-159548 active also in cells resistant to doxorubicin and with a multidrug resistance phenotype ( associated with MDR1 gene/P-glycoprotein overexpression ) , as well as in cells resistant to methotrexate or to cisplatin .', ARRAY("PNU-159548", "doxorubicin", "methotrexate", "cisplatin"), ARRAY()),
('Synergistic cytotoxic effects of recombinant human tumor necrosis factor and Etoposide ( VP16 ) or Doxorubicin on A2774 human epithelial ovarian cancer cell line .', ARRAY("human", "tumor", "necrosis", "factor", "Etoposide", "Doxorubicin"), ARRAY("VP16", "Doxorubicin", "A2774")),
('Many ESBL-producing E. coli had significantly lower susceptibility to gentamicin ( p ＜ 0.0001 ) and the quinolones nalidixic acid ( p＝0.004 ) and ciprofloxacin ( p ＜ 0.0001 ) than non-producers .', ARRAY("gentamicin", "nalidixic", "ciprofloxacin"), ARRAY("gentamicin", "nalidixic", "acid", "ciprofloxacin")),
('The resistance of topotecan in MDR HL-60 cells was potently reversed by the addition of amlodipine .', ARRAY("topotecan", "amlodipine"), ARRAY("Topotecan", "Amlodipine")),
('In separate studies , the electrocardiogram ( ECG ) and cardiovascular effects of loratadine ( 30 and 100 mg/kg , i.v . ) , terfenadine ( 10 mg/kg , i.v . ) , promethazine ( 5 mg/kg , i.v . ) and diphenhydramine ( 20 mg/kg , i.v . ) were evaluated .', ARRAY("loratadine", "terfenadine", "promethazine", "diphenhydramine"), ARRAY("loratadine", "terfenadine", "promethazine", "diphenhydramine")),
('[ Clinical evaluation of effects from neoadjuvant chemotherapy with epirubicin plus paclitaxel in cases of locally advanced breast cancer -- comparative study of treatment with 2 and 4 cycles ] .', ARRAY("epirubicin", "paclitaxel"), ARRAY("Epirubicin", "Paclitaxel")),
('When systemically active chemotherapy doses were reached , further dose escalation was discontinued , and a phase II dose-range was established ( pemetrexed 500 mg/m(2 ) and carboplatin AUC = 5 - 6 ) .', ARRAY("pemetrexed", "carboplatin"), ARRAY("Carboplatin", "Pemetrexed")),
('Increased dose density is feasible : a pilot study of adjuvant epirubicin and cyclophosphamide followed by paclitaxel , at 10- or 11-day intervals with filgrastim support in women with breast cancer .', ARRAY("epirubicin", "cyclophosphamide", "paclitaxel", "filgrastim"), ARRAY("Adriamycin", "Cyclophosphamide", "Paclitaxel", "Filgrastim")),
('Those who test positive are treated with combinations of the following agents : omeprazole , clarithromycin , amoxicillin , tetracycline , and metronidazole .', ARRAY("omeprazole", "clarithromycin", "amoxicillin", "tetracycline", "metronidazole"), ARRAY("omeprazole", "clarithromycin", "amoxicillin", "tetracycline", "metronidazole")),
('Patients can be treatment-naïve for mCRPC or on first-line androgen receptor-targeted therapy for mCRPC ( ie , abiraterone or enzalutamide ) without evidence of progression at enrolment , and with no prior chemotherapy for mCRPC .', ARRAY("abiraterone", "enzalutamide"), ARRAY("Abiraterone", "Enzalutamide")),
('CYP2D6 metabolizes other opioid analgesics , including tramadol , dihydrocodeine , oxycodone and hydrocodone , although they have been less systematically studied .', ARRAY("tramadol", "hydrocodone"), ARRAY("Tramadol", "Dihydrocodeine", "Oxycodone", "Hydrocodone")),
('Atracurium or vecuronium was given for intubation .', ARRAY("Atracurium", "vecuronium"), ARRAY("Atropine", "Vecuronium", "Atracurium")),
('The data show that knockdown of Rad51 or BRCA2 greatly sensitizes cells to DSBs and the induction of cell death following temozolomide and nimustine ( ACNU ) .', ARRAY("temozolomide", "nimustine"), ARRAY("Temozolomide", "Nimustine", "DSB", "BRCA2", "Rad51", "knockdown")),
('The systemic treatment in both studies consisted of a four-drug-regimen ( VACA = vincristine , actinomycin D , cyclophosphamide , and adriamycin ; or VAIA = vincristine , actinomycin D , ifosfamide , and adriamycin ) and a total number of four courses , each lasting nine weeks , was recommended by the protocol .', ARRAY("vincristine", "actinomycin", "cyclophosphamide", "adriamycin", "vincristine", "actinomycin", "ifosfamide", "adriamycin"), ARRAY("vincristine", "actinomycin", "D", "cyclophosphamide", "adriamycin")),
('Rationale , design , and baseline characteristics of a trial of prevention of cardiovascular and renal disease with fosinopril and pravastatin in nonhypertensive , nonhypercholesterolemic subjects with microalbuminuria ( the Prevention of REnal and Vascular ENdstage Disease Intervention Trial [ PREVEND IT ] ) .', ARRAY("fosinopril", "pravastatin"), ARRAY("Atorvastatin", "Fosinopril")),
('In the present study , rabbits prepared with chronic vascular cannulae were used to study the effects of nicotine administration on plasma corticosterone , catecholamine ( epinephrine , norepinephrine and dopamine ) and glucose responses to physical restraint stress .', ARRAY("nicotine", "epinephrine"), ARRAY("Epinephrine", "Norepinephrine", "Dopamine", "Glucocorticoids", "Nicotine")),
('Phase III , randomized , double-blind , multicenter trial comparing orteronel ( TAK-700 ) plus prednisone with placebo plus prednisone in patients with metastatic castration-resistant prostate cancer that has progressed during or after docetaxel-based therapy : ELM-PC 5 .', ARRAY("TAK-700", "prednisone", "prednisone"), ARRAY("Orteronel", "prednisone")),
('Five clinical trials with temozolomide or dacarbazine have been performed in metastatic colorectal cancer ( mCRC ) with selection based on methyl-specific PCR ( MSP ) testing with modest results .', ARRAY("temozolomide", "dacarbazine"), ARRAY("Temozolomide", "Dacarbazine")),
('Heparin infusion , followed by oral warfarin , is indicated for symptomatic thromboembolic disease as well as for asymptomatic patients with substantial proximal deep venous thrombosis or large pulmonary emboli .', ARRAY("Heparin", "warfarin"), ARRAY("Heparin", "Warfarin")),
('Fluoxetine and paroxetine are potent inhibitors of CYP2D6 and administration of these SSRIs reduces the clinical benefit of an anticancer drug , such as tamoxifen , by decreasing the formation of active metabolites of this drug .', ARRAY("Fluoxetine", "paroxetine", "tamoxifen"), ARRAY("Fluoxetine", "Paroxetine")),
('Epirubicin is an anthracyclin , analogous to doxorubicin , with a different toxicologic pattern .', ARRAY("Epirubicin", "doxorubicin"), ARRAY("Anthracyclin", "Doxorubicin")),
('A total of 460 patients were randomized into four 10-day therapeutic schemes ( 115 patients per group ): ( i ) standard OCA , omeprazole , clarithromycin and amoxicillin ; ( ii ) triple OLA , omeprazole , levofloxacin and amoxicillin ; ( iii ) sequential OACM , omeprazole plus amoxicillin for 5 days , followed by omeprazole plus clarithromycin plus metronidazole for 5 days ; and ( iv ) modified sequential OALM , using levofloxacin instead of clarithromycin .', ARRAY("omeprazole", "clarithromycin", "amoxicillin", "omeprazole", "levofloxacin", "amoxicillin", "omeprazole", "amoxicillin", "omeprazole", "clarithromycin", "metronidazole", "levofloxacin", "clarithromycin"), ARRAY()),
('Captopril in combination with HYZ significantly reduced BP compared with controls but T replacement increased BP and coronary collagen deposition in spite of HYZ and captopril treatment .', ARRAY("Captopril", "HYZ", "captopril"), ARRAY("Angiotensin-Converting", "Enzyme", "Inhibitors", "Angiotensin", "II", "Receptor", "Blockers", "Angiotensin", "II", "Receptor", "Antagonists", "ACE", "inhibitors", "Angiotensin", "II", "Receptor", "Antagonists", "Angiotensin", "II", "Receptor", "Blockers", "Angiotensin-Converting", "Enzyme", "Inhibitors")),
('Targeted therapy was administered to match the P1K3CA , c-MET , and SPARC and COX2 aberrations with sirolimus+ crizotinib and abraxane+ celecoxib .', ARRAY("sirolimus+", "crizotinib", "abraxane+", "celecoxib"), ARRAY("sirolimus", "crizotinib", "abraxane", "celecoxib", "c-MET", "P1K3CA", "SPARC")),
('EURAMOS-1 results do not support the addition of ifosfamide and etoposide to postoperative chemotherapy in patients with poorly responding osteosarcoma because its administration was associated with increased toxicity without improving event-free survival .', ARRAY("ifosfamide", "etoposide", "postoperative", "chemotherapy"), ARRAY("Ifosfamide", "Etoposide")),
('The focus of this study is to investigate , whether altered expression levels of potentially relevant microRNAs ( miRs ) in serum are associated with response to trastuzumab or lapatinib .', ARRAY("trastuzumab", "lapatinib"), ARRAY("Trastuzumab", "Lapatinib", "Bevacizumab", "Cetuximab", "Erlotinib", "Sunitinib", "Panitumumab", "Cetuximab", "Bevacizumab", "Lapatinib", "Trastuzumab", "Erlotinib", "Panitumumab")),
('These results suggest that the addition of intravenous ceftriaxone during the first 3 days of hospitalization does not improve the cost-efficacy of oral norfloxacin in the prevention of bacterial infections in cirrhotic patients with gastrointestinal bleeding and high risk of infection .', ARRAY("ceftriaxone", "norfloxacin"), ARRAY("ceftriaxone", "norfloxacin")),
('However , clindamycin possesses only one of the two mechanisms of lincomycin action , which is bacteriostatic , against Escherichia coli .', ARRAY("clindamycin", "lincomycin"), ARRAY("Clindamycin")),
('This study evaluated the response rate of the combination therapy of aprinocarsen , gemcitabine , and carboplatin in previously untreated patients with advanced non-small cell lung cancer ( NSCLC ) .', ARRAY("aprinocarsen", "gemcitabine", "carboplatin"), ARRAY("Aprinocar", "Gemcitabine", "Carboplatin")),
('All patients were treated by surgical debridement followed by a combination of antibiotics ; ( ceftazidime , amoxy-clavulanic acid , co-trimoxazole and doxycycline ) for six months except for one who died due to fulminant septicemia .', ARRAY("ceftazidime", "amoxy-clavulanic", "co-trimoxazole", "doxycycline"), ARRAY("ceftazidime", "amoxicillin-clavulanate", "co-trimoxazole", "doxycycline")),
('To evaluate the eradication of Helicobacter pylori by therapy with a combination of 60 mg lansoprazole and 800 mg clarithromycin .', ARRAY("lansoprazole", "clarithromycin"), ARRAY()),
('All patients were treated with vincristine , doxorubicin , cyclophosphamide and actinomycin-D , alternating with ifosfamide and etoposide every 3 weeks .', ARRAY("vincristine", "doxorubicin", "cyclophosphamide", "actinomycin-D", "ifosfamide", "etoposide"), ARRAY("vincristine", "doxorubicin", "cyclophosphamide", "actinomycin-D", "ifosfamide", "etoposide")),
('We identified 49 and 220 patients treated with sorafenib and sunitinib , respectively , as first-line therapy in the Asan Medical Centre from April 2005 to March 2011 .', ARRAY("sorafenib", "sunitinib"), ARRAY("Aspirin", "Sunitinib", "Sorafenib")),
('We studied the combination of pemetrexed , a multi-targeted antifolate , and cetuximab , an mAb against the epidermal growth factor receptor , with radiotherapy in poor prognosis head and neck cancer .', ARRAY("pemetrexed", "cetuximab"), ARRAY("pemetrexed", "cetuximab")),
('Boosting darunavir with ritonavir instead of with cobicistat may be preferred if darunavir is to be combined with etravirine in clinical practice .', ARRAY("darunavir", "ritonavir", "cobicistat", "darunavir", "etravirine"), ARRAY("Boosting", "darunavir", "with", "ritonavir", "instead", "of", "with", "cobicistat", "may", "be", "preferred", "if", "darunavir", "is", "to", "be", "combined", "with", "etravirine", "in", "clinical", "practice")),
('Single-agent paclitaxel and vinorelbine are recommended treatments for advanced breast cancer ( ABC ) non-responsive to hormone therapy and without visceral crisis .', ARRAY("paclitaxel", "vinorelbine"), ARRAY("Paclitaxel", "Vinorelbine")),
('Nal-IRI with 5-fluorouracil ( 5-FU ) and leucovorin or gemcitabine plus cisplatin in advanced biliary tract cancer - the NIFE trial ( AIO-YMO HEP-0315 )', ARRAY("5-fluorouracil", "leucovorin", "gemcitabine", "cisplatin"), ARRAY("Nal-IRI", "with", "5-fluorouracil", "leucovorin", "gemcitabine", "cisplatin", "AIO-YMO", "HEP-0315")),
('Ten patients were withdrawn from the terbutaline group because treatment was insufficiently effective , whereas only one dropped out of the budesonide group .', ARRAY("terbutaline", "budesonide"), ARRAY("Terbutaline", "Budesonide")),
('We conducted a double-blind cross-over study in ten volunteers aged from 19 to 30 years , to compare the pain control effects of a single oral dose of two analgesic compounds ( drug A : propyphenazone mg 250 , ethylmorphine mg 5 , caffeine mg 5 ; drug B : dipyrone mg 500 , diphenhydramine mg 12.5 , adiphenine mg 5 , ethyl aminobenzoate mg 2.5 ) in an experimental pain model using stimulation of dental pulp .', ARRAY("propyphenazone", "ethylmorphine", "caffeine", "dipyrone", "diphenhydramine", "adiphenine", "aminobenzoate"), ARRAY("Propyphenazone", "Ethylmorphine", "Caffeine", "Dipyrone", "Diphenhydramine", "Adiphenine", "Ethyl", "aminobenzoate")),
('Forty-three mCRC patients who received cetuximab or panitumumab between April 2012 and December 2015 were the subjects of the present study .', ARRAY("cetuximab", "panitumumab"), ARRAY("cetuximab", "panitumumab")),
('Furthermore , 11 AML patients at primary diagnosis , including five AML patients with P-gp overexpression , who were treated with idarubicin , vepesid , and cytarabine V ( ara-C ) showed a complete remission .', ARRAY("idarubicin", "vepesid", "cytarabine"), ARRAY("Idarubicin", "Vepesid", "Cytarabine", "(Ara-C)")),
('Phase Ib Study of Bavituximab With Carboplatin and Pemetrexed in Chemotherapy-Naive Advanced Nonsquamous Non-Small-Cell Lung Cancer .', ARRAY("Bavituximab", "Carboplatin", "Pemetrexed"), ARRAY()),
('In this multicenter study , the reliability of two nonradiometric , fully automated systems , the MB/BacT and BACTEC MGIT 960 systems , for testing the susceptibilities of 82 Mycobacterium tuberculosis strains to isoniazid , rifampin , ethambutol , and streptomycin was evaluated in comparison with the radiometric BACTEC 460 TB system .', ARRAY("isoniazid", "rifampin", "ethambutol", "streptomycin"), ARRAY("Isoniazid", "Rifampin", "Ethambutol", "Streptomycin", "MB/BacT", "BACTEC", "MGIT", "960")),
('Most frequent first line therapy was Sunitinib ( 66 % ) , followed by Sorafenib ( 20 % ) and Pazopanib ( 10 % ) .', ARRAY("Sunitinib", "Sorafenib", "Pazopanib"), ARRAY("Most", "frequent", "first", "line", "therapy", "Sorafenib", "Pazopanib")),
('Treatment of unresectable GISTs involves systemic chemotherapy with tyrosine kinase inhibitors , imatinib and sunitinib being first-line and second-line drugs .', ARRAY("imatinib", "sunitinib"), ARRAY("tumors", "tyrosine", "kinase", "inhibitors", "imatinib", "sunitinib")),
('Such therapy has included weekly paclitaxel in combination with carboplatin/cisplatin plus topotecan , and carboplatin plus doxorubicin .', ARRAY("paclitaxel", "topotecan", "carboplatin", "doxorubicin", "carboplatin/cisplatin"), ARRAY("Paclitaxel", "Carboplatin", "Cisplatin", "Topotecan", "Doxorubicin")),
('This phase II study was performed to assess the efficacy and safety of the combination regimen of temozolomide and docetaxel in patients with advanced metastatic melanoma .', ARRAY("temozolomide", "docetaxel"), ARRAY("Temozolomide", "Docetaxel")),
('Addition of verapamil and tamoxifen to the initial chemotherapy of small cell lung cancer .', ARRAY("verapamil", "tamoxifen"), ARRAY("tamoxifen", "verapamil")),
('In a scenario analysis comparing pembrolizumab with ipilimumab , the estimated ICER was USD8,904 .', ARRAY("pembrolizumab", "ipilimumab"), ARRAY("Ipilimumab", "Pembrolizumab")),
('Long-term cardiac outcomes of patients with HER2-positive breast cancer treated in the adjuvant lapatinib and/or trastuzumab Treatment Optimization Trial .', ARRAY("lapatinib", "trastuzumab"), ARRAY("Herceptin", "Trastuzumab", "Lapatinib")),
('To determine the current status of ivermectin , abamectin and praziquantel combined , and fenbendazole resistance to Parascaris spp . in horses in Saudi Arabia .', ARRAY("ivermectin", "abamectin", "praziquantel", "fenbendazole"), ARRAY("Abamectin", "Ivermectin", "Praziquantel", "Fenbendazole")),
('To determine the response rate ( RR ) and survival produced by carboplatin + gemcitabine therapy in patients with untreated extensive small cell lung cancer ( ESCLC ) .', ARRAY("carboplatin", "gemcitabine"), ARRAY()),
('The interaction of chloroquine and citalopram in vitro resulted in a synergistic response in the chloroquine-resistant strain but there was no interaction between the drugs in the chloroquine-sensitive strain -- a pattern found with other reversal agents .', ARRAY("chloroquine", "citalopram"), ARRAY("Chloroquine", "Citalopram")),
('Interfacial Phenomenon Based Biocompatible Alginate-Chitosan Nanoparticles Containing Isoniazid and Pyrazinamide .', ARRAY("Isoniazid", "Pyrazinamide"), ARRAY("Interfacial", "Phenomenon", "Based", "Biocompatible", "Alginate-Chitosan", "Nanoparticles", "Containing", "Isoniazid", "and", "Pyrazinamide")),
('In a 27-day inpatient study , 10 methamphetamine-dependent individuals participated in a double-blind , placebo-controlled , cross-over design , with oral doses of topiramate ( 0 , 100 , and 200 mg ) administered as a pretreatment before intravenous doses of methamphetamine ( 0 , 15 , and 30 mg ) .', ARRAY("topiramate", "methamphetamine"), ARRAY()),
('Synergism between penicillin , clindamycin , or metronidazole and gentamicin against species of the Bacteroides melaninogenicus and Bacteroides fragilis groups .', ARRAY("penicillin", "clindamycin", "metronidazole", "gentamicin"), ARRAY("Penicillin", "clindamycin", "metronidazole", "gentamicin")),
('Thirty-one animals were allocated randomly to three groups , all administered four boluses of 0.25 mg/kg rTPA every 10 min for 30 min , 17 mg/kg aspirin intravenously , and heparin ( as a 100 IU/kg bolus followed by infusion of 50 IU/kg heparin per h ) , hirudin ( as a 2 mg/kg bolus followed by infusion of 1 mg/kg hirudin per h ) , or Yagin ( as an 80 micrograms/kg bolus followed by infusion of 43 micrograms/kg Yagin per h ) .', ARRAY("aspirin", "heparin", "heparin"), ARRAY("0.25", "mg/kg", "rTPA", "17", "mg/kg", "aspirin", "100", "IU/kg", "heparin", "50", "IU/kg", "hirudin", "80", "micrograms/kg", "Yagin")),
('In vivo evofosfamide was tumor suppressive as a single agent and cooperated with paclitaxel to reduce mammary tumor growth .', ARRAY("evofosfamide", "paclitaxel"), ARRAY("Evofosfamide", "Paclitaxel")),
('Paclitaxel and vinorelbine are among the most active new agents in metastatic breast cancer .', ARRAY("Paclitaxel", "vinorelbine"), ARRAY("taxolome", "paclitaxel", "vinorelbine")),
('Treatments included recombinant human leptin ( 10 - 100nM ) , recombinant human IL-6 ( 0.3 - 3nM ) , or recombinant human erythropoietin ( Epo ) ( 10mU/ml ) .', ARRAY("leptin", "erythropoietin"), ARRAY("Recombinant", "human", "leptin", "recombinant", "human", "IL-6", "recombinant", "human", "erythropoietin")),
('Lenalidomide is currently being tested in combination with both standard and novel agents , including bortezomib , for patients with relapsed/refractory multiple myeloma .', ARRAY("Lenalidomide", "bortezomib"), ARRAY("Lenalidomide", "Bortezomib")),
('[ Efficacy of fluvoxamine combined with extended-release methylphenidate on treatment-refractory obsessive-compulsive disorder ] .', ARRAY("fluvoxamine", "methylphenidate"), ARRAY("Efficacy", "of", "fluvoxamine", "combined", "with", "extended-release", "methylphenidate", "on", "treatment-refractory", "obsessive-compulsive", "disorder")),
('Analysis of HER Family ( HER1 - 4 ) Expression as a Biomarker in Combination Therapy with Pertuzumab , Trastuzumab and Docetaxel for Advanced HER2-positive Breast Cancer .', ARRAY("Pertuzumab", "Trastuzumab", "Docetaxel"), ARRAY()),
('Hepatic arterial infusion of floxuridine and systemic administration of gemcitabine and oxaliplatin .', ARRAY("floxuridine", "gemcitabine", "oxaliplatin"), ARRAY("Fluorouracil", "Gemcitabine", "Oxaliplatin")),
('Data from the New Zealand Intensive Medicines Monitoring Programme indicate that celecoxib 200 mg/day and rofecoxib 25 mg/day are/were the most commonly prescribed doses and that 6 % of patients had taken rofecoxib 50 mg/day for longer than recommended .', ARRAY("celecoxib", "rofecoxib", "rofecoxib"), ARRAY("celecoxib", "rofecoxib", "celecoxib", "200", "mg/day", "rofecoxib", "25", "mg/day")),
('Chemoimmunotherapy with cyclophosphamide , doxorubicin , vincristine , and prednisolone combined with rituximab ( R-CHOP ) is currently the first-line therapy for diffuse large B-cell lymphoma ( DLBCL ) .', ARRAY("cyclophosphamide", "doxorubicin", "vincristine", "prednisolone", "rituximab"), ARRAY("Chemoimmunotherapy", "with", "cyclophosphamide", "doxorubicin", "vincristine", "and", "prednisolone", "combined", "with", "rituximab", "(R-CHOP)")),
('The purpose of our study was to compare the survival of porcine lung allografts after induction with either cyclosporine A ( CsA ) or tacrolimus .', ARRAY("cyclosporine", "tacrolimus"), ARRAY("cyclosporine", "A", "tacrolimus")),
('Cystoid Macular Edema during Treatment with Paclitaxel and Bevacizumab in a Patient with Metastatic Breast Cancer : A Case Report and Literature Review .', ARRAY("Paclitaxel", "Bevacizumab"), ARRAY("Paclitaxel", "Bevacizumab")),
('The results suggest that third-line chemotherapy with combined bevacizumab and S-1 is safe and may delay the progression of mCRC resistant to oxaliplatin and irinotecan with mutated KRAS .', ARRAY("bevacizumab", "S-1", "oxaliplatin", "irinotecan"), ARRAY("Bevacizumab", "S-1", "oxaliplatin", "irinotecan", "KRAS")),
('In accordance with their different pharmacological profiles , the three NLs iloperidone , clozapine , and haloperidol have different effects in this preclinical cognitive task .', ARRAY("iloperidone", "clozapine", "haloperidol"), ARRAY("Iloperidone", "Clozapine", "Haloperidol")),
('In the comparison between rivaroxaban-based triple therapy and ticagrelor + aspirin , the RR was 1 and its 95 % CI remained within a post-hoc margin of ± 15 % .', ARRAY("ticagrelor", "aspirin"), ARRAY("Rivaroxaban", "Ticagrelor", "Aspirin")),
('The incidence of cross-resistance between indinavir , nelfinavir , ritonavir and saquinavir was high ( 60 - 90 % ) .', ARRAY("indinavir", "nelfinavir", "ritonavir", "saquinavir"), ARRAY("indinavir", "nelfinavir", "ritonavir", "saquinavir")),
('Maintenance Treatment With Low-Dose Mercaptopurine in Combination With Allopurinol in Children With Acute Lymphoblastic Leukemia and Mercaptopurine-Induced Pancreatitis .', ARRAY("Mercaptopurine", "Allopurinol"), ARRAY("Allopurinol", "Azathioprine", "Mercaptopurine", "Mercaptopurine-Induced", "Pancreatitis")),
('The aim of this study was to assess the predictive and prognostic value of clinical response to second line treatment ( with capecitabine or with a two-drug regimen including irinotecan ) and to analyze its relation to selected clinical and pathological variables with respect to time to disease progression .', ARRAY("capecitabine", "irinotecan"), ARRAY("Capecitabine", "Irinotecan")),
('An 8-week , randomized , parallel-group , double-blind international trial comparing the once-daily single-pill combination of telmisartan 80 mg and amlodipine 10 mg ( T/A ; n = 352 ) with once-daily amlodipine 10 mg ( A ; n = 354 ) in patients with type 2 diabetes mellitus and stage 1 or 2 hypertension ( systolic BP [ SBP ] > 150 mm Hg ) .', ARRAY("telmisartan", "amlodipine", "amlodipine"), ARRAY("telmisartan", "amlodipine")),
('Pharmacokinetic studies evaluating the concurrent use of tenofovir and didanosine have been performed in healthy volunteers .', ARRAY("tenofovir", "didanosine"), ARRAY("Didanosine", "Tenofovir")),
('This comparative phase III trial of mitoxantrone+vinorelbine ( MV ) versus 5-fluorouracil+cyclophosphamide+either doxorubicin or epirubicin ( FAC/FEC ) in the treatment of metastatic breast cancer was conducted to determine whether MV would produce equivalent efficacy , while resulting in an improved tolerance in relation to alopecia and nausea/vomiting .', ARRAY("doxorubicin", "epirubicin"), ARRAY("Mitoxantrone", "Vinorelbine", "5-fluorouracil", "Cyclophosphamide", "Doxorubicin", "Epirubicin", "FAC", "FEC")),
('Overnight urinary oxytocin and vasopressin levels were obtained from 62 healthy males ( age range : 18 - 26 years ) to compare with trait measures of trust and aggressive behavior .', ARRAY("oxytocin", "vasopressin"), ARRAY("Oxytocin", "Vasopressin")),
('Additionally , the combined effect of Lapatinib together with Herceptin or Cetuximab on cell-mediated cytotoxicity was evaluated .', ARRAY("Lapatinib", "Herceptin", "Cetuximab"), ARRAY("Herceptin", "Lapatinib")),
('In a case-control study , we compared 52 consecutive patients undergoing isolated CABG on aspirin and clopidogrel 75mg/d versus 50 controls on aspirin monotherapy .', ARRAY("aspirin", "clopidogrel", "aspirin"), ARRAY("Aspirin", "Clopidogrel", "Aspirin", "monotherapy", "Clopidogrel", "monotherapy")),
('The MBCs were 1 to 2 tubes higher than the broth dilution MICs for levofloxacin , 1 to 3 tubes higher than the broth dilution MICs for ofloxacin , 1 to 3 tubes higher than the broth dilution MICs for erythromycin , and the same as the broth dilution MICs for rifampin .', ARRAY("levofloxacin", "erythromycin", "rifampin"), ARRAY("Levofloxacin", "ofloxacin", "erythromycin", "rifampin")),
('Ketanserin ( a 5-HT2 antagonist , 10 mumol/L ) and tropisetron ( a 5-HT3 antagonist , 1 mumol/L ) had no effect .', ARRAY("Ketanserin", "tropisetron"), ARRAY("Ketanserin", "Tropisetron")),
('This three-phase study was designed to determine if a pharmacokinetic drug-drug interaction exists between zidovudine and oxazepam .', ARRAY("zidovudine", "oxazepam"), ARRAY("Zidovudine", "Oxazepam")),
('Patients with apical CFTR protein showed higher residual chloride secretion than those without ( amiloride to isoprenaline value of 4.59 and 0.56 mV , respectively , p = 0.01 ) .', ARRAY("amiloride", "isoprenaline"), ARRAY("Amiloride", "Isoprenaline")),
('African American men were more likely to receive ketoconazole than abiraterone , enzalutamide , or docetaxel ( AME , 2.8 % ; 95 % CI , 0.7%-4.9 % ) .', ARRAY("ketoconazole", "abiraterone", "enzalutamide", "docetaxel"), ARRAY("Abiraterone", "Ketoconazole", "Enzalutamide", "Docetaxel")),
('These observations are important in the development of vorinostat , and may have clinical implications on other cancer and noncancer drugs that are UGT2B17 substrates such as exemestane and ibuprofen .', ARRAY("vorinostat", "exemestane", "ibuprofen"), ARRAY()),
('Additional results from MTT cell viability assays demonstrated that H1975 cell proliferation was not significantly decreased after Wnt inhibition by XAV939 , but combination treatment with everolimus ( mTOR inhibitor ) and erlotinib resulted in synergistic cell growth inhibition .', ARRAY("XAV939", "everolimus", "erlotinib"), ARRAY("Avastin", "Erlotinib", "Everolimus", "XAV939")),
('The addition of dexmedetomidine to bupivacaine 0.5 % in EUS-CPN demonstrated beneficial effects as regards the degree and duration of pain relieve with negligible effect on the patient survival .', ARRAY("dexmedetomidine", "bupivacaine"), ARRAY("Dexmedetomidine", "Bupivacaine", "0.5%")),
('The incidence of carcinoma , confirmed microscopically , was : control 14/20 ( 70 % ) ; high-dose gefitinib , 7/20 ( 35 % ) ; low-dose gefitinib , 7/20 ( 35 % ) ; high-dose meloxicam 7/21 ( 33 % ) ; and low-dose meloxicam , 12/20 ( 60 % ) .', ARRAY("gefitinib", "gefitinib", "meloxicam", "meloxicam"), ARRAY("carcinoma", "gefitinib", "high-dose", "gefitinib", "low-dose", "gefitinib", "meloxicam", "high-dose", "meloxicam", "low-dose", "meloxicam")),
('Patients with incurable cancer causing chronic pain rated above 6/10 on a numerical scale while receiving high-dose opioid therapy ( more than 200 mg/d of oral morphine equivalent ) and/or exhibiting severe opioid-related adverse events received intrathecal infusions of ziconotide combined with morphine , ropivacaine , and clonidine .', ARRAY("ziconotide", "morphine", "ropivacaine", "clonidine"), ARRAY()),
('Both adalimumab and etanercept were more effective than methotrexate in slowing radiographic joint damage .', ARRAY("adalimumab", "etanercept", "methotrexate"), ARRAY("Adalimumab", "Etanercept")),
('Combination chemotherapy with gemcitabine ( Gem ) , doxorubicin ( Dox ) , and paclitaxel ( Pac ) ( GAT ) has been considered attractive as first-line treatment in metastatic breast cancer .', ARRAY("gemcitabine", "doxorubicin", "paclitaxel"), ARRAY("Gemcitabine", "Doxorubicin", "Paclitaxel")),
('Anti-cancer agents like adriamycin , mitomycin-C , bleomycin , and etoposide express their cell-killing activity partly through oxygen radicals .', ARRAY("adriamycin", "mitomycin-C", "bleomycin", "etoposide"), ARRAY("Adriamycin", "Mitomycin-C", "Bleomycin", "Etoposide")),
('The efficacy and safety of isepamicin compared with amikacin in the treatment of intra-abdominal infections .', ARRAY("isepamicin", "amikacin"), ARRAY()),
('Imipenem was compared with nafcillin and with penicillin plus gentamicin in the therapy of experimental endocarditis induced in rabbits by Staph . aureus and Str . faecalis , respectively .', ARRAY("Imipenem", "nafcillin", "penicillin", "gentamicin"), ARRAY("Imipenem", "Nafcillin", "Penicillin", "Gentamicin")),
('The total dose of paclitaxel ( 175 - 200 mg/m2 ) ; cisplatin ( 75 mg/m2 ) ; and etoposide ( 175 - 200 mg/m2 ) was divided into five daily doses administered over 3 h with cycles repeated at 21 - 28 days .', ARRAY("paclitaxel", "cisplatin", "etoposide"), ARRAY("Paclitaxel", "Cisplatin", "Etoposide")),
('In this phase 3 trial , we randomly assigned ( in a 1:1:1 ratio ) patients with advanced renal cell carcinoma and no previous systemic therapy to receive lenvatinib ( 20 mg orally once daily ) plus pembrolizumab ( 200 mg intravenously once every 3 weeks ) , lenvatinib ( 18 mg orally once daily ) plus everolimus ( 5 mg orally once daily ) , or sunitinib ( 50 mg orally once daily , alternating 4 weeks receiving treatment and 2 weeks without treatment ) .', ARRAY("lenvatinib", "pembrolizumab", "lenvatinib", "everolimus", "sunitinib"), ARRAY("Lenvatinib", "Pembrolizumab", "Everolimus")),
('Only chemotherapy by intravenous administration of 2 courses of 120 - 150 mg ACNU ( 1.7 - 2.2 mg/kg ) and 4 mg vincristine ( 0.06 mg/kg ) with intrathecal administration of methotrexate was given at this time .', ARRAY("ACNU", "vincristine", "methotrexate"), ARRAY("ACNU", "Methotrexate")),
('In first-line treatment of intermediate- to poor-risk patients , the CheckMate 214 study demonstrated a significant survival advantage for nivolumab and ipilimumab versus sunitinib .', ARRAY("nivolumab", "ipilimumab", "sunitinib"), ARRAY("nivolumab", "ipilimumab", "sunitinib")),
('In addition , docetaxel has a longer retention time in tumor cells than paclitaxel because of greater uptake and slower efflux .', ARRAY("docetaxel", "paclitaxel"), ARRAY("Docetaxel", "Paclitaxel")),
('The total sample size was 103 , consisting of 31 , 23 , and 19 patients in olanzapine , risperidone , and clozapine groups , respectively and 30 controls .', ARRAY("olanzapine", "risperidone", "clozapine"), ARRAY("Olanzapine", "Risperidone", "Clozapine")),
('The relative analgesic potency ratios were 0.65 ( 0.56 - 0.76 ) for ropivacaine : bupivacaine , 0.80 ( 0.70 - 0.92 ) for ropivacaine : levobupivacaine , and 0.81 ( 0.69 - 0.94 ) for levobupivacaine : bupivacaine .', ARRAY("ropivacaine", "bupivacaine", "ropivacaine", "levobupivacaine", "levobupivacaine", "bupivacaine"), ARRAY("bupivacaine", "ropivacaine", "levobupivacaine")),
('Lapatinib ( LPT ) could sensitize human epidermal growth factor receptor-2 ( HER-2 ) positive breast cancer to paclitaxel ( PTX ) and induce synergetic action with PTX in preclinical test and phase II/III trial .', ARRAY("Lapatinib", "paclitaxel"), ARRAY()),
('Two of nine patients received 15 mg/kg bevacizumab IV , 90 mg/m(2 ) irinotecan orally for five consecutive days , 100 mg/m(2)/day temozolomide IV for 5 days , and 1.5 mg/m(2 ) vincristine IV , each administered every 21 days .', ARRAY("bevacizumab", "irinotecan", "temozolomide", "vincristine"), ARRAY("Bevacizumab", "Irinotecan", "Temozolomide", "Vincristine")),
('These findings may help clinicians identify patients for whom acamprosate and naltrexone may be most beneficial .', ARRAY("acamprosate", "naltrexone"), ARRAY()),
('From August 2002 to August 2004 , 42 patients with metastatic breast cancer were recruited for treatment with pegylated liposomal doxorubicin 40 mg/m(2 ) intravenously ( i.v . ) on day 1 and vinorelbine 30 mg/m(2 ) i.v . on days 1 and 15 every 4 weeks .', ARRAY("doxorubicin", "vinorelbine"), ARRAY("doxorubicin", "vinorelbine")),
('The aim of the present study was to investigate the combined effects of simvastatin and exemestane on MCF-7 human breast cancer cells .', ARRAY("simvastatin", "exemestane"), ARRAY("Simvastatin", "exemestane")),
('Synergy of Omeprazole and Praziquantel In Vitro Treatment against Schistosoma mansoni Adult Worms .', ARRAY("Omeprazole", "Praziquantel"), ARRAY("Omeprazole", "Praziquantel")),
('Inclusion criteria were proven infection with evidence of a bacterial strain of PA resistant to all β-lactams and treated with the association of at least aztreonam plus cefepime .', ARRAY("aztreonam", "cefepime"), ARRAY("Aztreonam", "Cefepime")),
('Comparison of the additive effects of nipradilol and carteolol to latanoprost in open-angle glaucoma .', ARRAY("nipradilol", "carteolol", "latanoprost"), ARRAY("nipradilol", "carteolol", "latanoprost")),
('The poly(ADP-ribose ) polymerase ( PARP ) inhibitor olaparib has recently received approval from the Food and Drug Administration ( FDA ) and European Medicines Agency ( EMA ) , with a second agent ( rucaparib ) likely to be approved in the near future .', ARRAY("olaparib", "rucaparib"), ARRAY("Olaparib", "Rucaparib")),
('Fluvoxamine is a fairly potent inhibitor of CYP2C19 and it has the potential for causing drug-drug interactions with substrates for CYP2C19 such as imipramine , clomipramine , amitriptyline and diazepam .', ARRAY("Fluvoxamine", "imipramine", "clomipramine", "amitriptyline", "diazepam"), ARRAY("Fluvoxamine", "Imipramine", "Clomipramine", "Amitriptyline", "Diazepam")),
('We have compared amphotericin B , flucytosine , ketoconazole and fluconazole susceptibilities of 40 clinical isolates of Candida albicans by broth microdilution in two different media : RPMI 1640 ( RPMI ) and the same medium supplemented with 18 g of glucose per L ( RPMI-2 % glucose ) .', ARRAY("amphotericin", "flucytosine", "ketoconazole", "fluconazole"), ARRAY("Amphotericin", "B", "Flucytosine", "Ketoconazole", "Fluconazole")),
('Over the past decade , paclitaxel , docetaxel , vinorelbine , gemcitabine , irinotecan , and topotecan have been introduced into the clinic .', ARRAY("paclitaxel", "docetaxel", "vinorelbine", "gemcitabine", "irinotecan", "topotecan"), ARRAY("Paclitaxel", "Docetaxel", "Vinorelbine", "Gemcitabine", "Irinotecan", "Topotecan")),
('Alternatively , it may be that the adverse effects of pindolol and propranolol are due to the simultaneous blockade of both beta 1- and beta 2-adrenoceptors .', ARRAY("pindolol", "propranolol"), ARRAY()),
('The combinations of irinotecan and mitomycin C or oxaliplatin have given very good results with high objective response rates and good tolerance .', ARRAY("irinotecan", "mitomycin", "oxaliplatin"), ARRAY("Irinotecan", "Mitomycin", "C", "Oxaliplatin")),
('The NHB of strategy A was 31.6 QALMs versus strategy C when palbociclib was included in strategy C ; similarly , strategy A had a NHB of 13.8 QALMs versus strategy C when pertuzumab was included in strategy C. Monte-Carlo simulation demonstrated that the main factor influencing NHB of strategy A over strategy C was the cost of systemic therapy .', ARRAY("palbociclib", "pertuzumab"), ARRAY("Palbociclib", "Pertuzumab")),
('In the second study 12 patients with small cell undifferentiated cancers were treated with carboplatin , etoposide and ifosfamide .', ARRAY("carboplatin", "etoposide", "ifosfamide"), ARRAY("Carboplatin", "Etoposide", "Ifosfamide")),
('3 . Systemic ( i.v . ) pretreatment with furosemide ( 2 - 10 mg kg-1 ) increased urine volume and dose-dependently inhibited the pressor response to i.c.v . clonidine ( 10 micrograms ) , and a long-lasting depressor response to clonidine was observed .', ARRAY("furosemide", "clonidine", "clonidine"), ARRAY("Clonidine", "Furosemide")),
('Nine studies connected to the network for the PFS analysis in which necitumumab in combination with gemcitabine and cisplatin was associated with the lowest HR .', ARRAY("necitumumab", "gemcitabine", "cisplatin"), ARRAY()),
('To evaluate the efficacy and toxicities of gemcitabine combined with ifosfamide and anthracycline chemotherapy for recurrent platinum resistant ovarian epithelial cancer .', ARRAY("gemcitabine", "ifosfamide", "anthracycline"), ARRAY()),
('Combination of erlotinib and sorafenib , synergistic in GSC11 , induced apoptosis and autophagic cell death associated with suppressed Akt and ERK signaling pathways and decreased nuclear PKM2 and β-catenin in vitro , and tended to improve survival of nude mice bearing GSC11 brain tumor .', ARRAY("erlotinib", "sorafenib"), ARRAY("Erlotinib", "Sorafenib")),
('In the first trial , paclitaxel and doxorubicin were alternated every 3 weeks in doses of 200 mg/m2 and 75 mg/m2 , respectively , for patients who had received no more than one prior chemotherapeutic regimen .', ARRAY("paclitaxel", "doxorubicin"), ARRAY("Paclitaxel", "Doxorubicin")),
('Thirty-one patients with histologically proven small cell lung cancer were treated with cyclophosphamide 1 g m-2 and etoposide ( VP16 - 213 ) 125 mg m-2 both intravenously on day 1 followed by etoposide 250 mg m-2 orally on days 2 - 3 for a maximum of six courses at 3 weekly intervals .', ARRAY("cyclophosphamide", "etoposide", "etoposide"), ARRAY("Cyclophosphamide", "Etoposide", "VP-16", "213", "Etoposide", "250", "mg")),
('Pemetrexed plus carboplatin followed by pemetrexed , docetaxel , atezolizumab and S-1 were performed in sequence .', ARRAY("Pemetrexed", "carboplatin", "pemetrexed", "docetaxel", "atezolizumab"), ARRAY("Pemetrexed", "Carboplatin", "Pemetrexed", "Docetaxel", "Atezolizumab", "S-1")),
('The initial response ( IR ) was sustained for a mean ( s.d . ) of 309 ( 244 ) days with vildagliptin versus 270 ( 223 ) days for glimepiride ( p < 0.001 ) ( IR = nadir HbA1c where change from baseline > or = 0.5 % or HbA1c < or = 6.5 % within the first six months of treatment .', ARRAY("vildagliptin", "glimepiride"), ARRAY()),
('The aim of this phase I study was to determine the maximum tolerated dose of a 3-h infusion of paclitaxel , combined with carboplatin at a fixed AUC of 7 mg ml-1 min every 4 weeks for up to six cycles and to evaluate any possible pharmacokinetic interaction .', ARRAY("paclitaxel", "carboplatin"), ARRAY("Paclitaxel", "Carboplatin")),
('Efficacy of tramadol in combination with doxepin or venlafaxine in inhibition of nociceptive process in the rat model of neuropathic pain : an isobolographic analysis .', ARRAY("tramadol", "doxepin", "venlafaxine"), ARRAY("Tramadol", "Doxepin", "Venlafaxine")),
('All those treated with mefloquine plus artemether survived and their parasite clearance time and fever clearance time were significantly shorter than those of patients receiving quinine .', ARRAY("mefloquine", "artemether", "quinine"), ARRAY("Artemether", "Mefloquine")),
('Currently available adrenal steroidogenesis inhibitors , including ketoconazole , metyrapone , etomidate , and mitotane , have variable efficacy and significant side effects , and none are approved by the US Food and Drug Administration for CS .', ARRAY("ketoconazole", "metyrapone", "etomidate", "mitotane"), ARRAY("Ketoconazole", "Metyrapone", "Etomidate", "Mitotane")),
('The differential effects of haloperidol and methamphetamine on time estimation in the rat .', ARRAY("haloperidol", "methamphetamine"), ARRAY("Haloperidol", "Methamphetamine")),
('In the present study , we evaluated the efficacy and safety of the weekly combination of etoposide , leucovorin ( LV ) and 5-fluorouracil ( 5-FU ) when administered as second-line chemotherapy in patients with relapsed/refractory advanced colorectal cancer ( ACC ) , previously treated with weekly LV+5-FU .', ARRAY("etoposide", "leucovorin", "5-fluorouracil"), ARRAY("etoposide", "leucovorin", "(LV)", "5-fluorouracil", "(5-FU)")),
('Diazoxide , nifedipine and 2-deoxy glucose suppressed ( p < 0.05 ) glucose stimulated insulin secretion in AtT20HI-GLUT2-GK-6 cells .', ARRAY("Diazoxide", "nifedipine", "2-deoxy", "glucose"), ARRAY("Diabetes", "medications", "nifedipine", "diazoxide")),
('Reducing the duration of medication use postoperatively may also minimize the possible side effects of ketorolac and codeine , which could develop with extended periods of use .', ARRAY("ketorolac", "codeine"), ARRAY()),
('Lenalidomide plus dexamethasone is a reference treatment for relapsed multiple myeloma .', ARRAY("Lenalidomide", "dexamethasone"), ARRAY("Lenalidomide", "Dexamethasone")),
('Following definitive treatment , patients were randomized to either cyclophosphamide 1 g/m2 intravenously every 3 weeks for 2 years , estramustine phosphate 600 mg/m2 orally daily for 2 years or to observation only .', ARRAY("cyclophosphamide", "estramustine"), ARRAY("Cyclophosphamide", "Estramustine", "phosphate")),
('The authors report on anemia observed during preoperative paclitaxel and carboplatin chemotherapy in patients with advanced head and neck carcinoma and discuss how the use of recombinant human erythropoietin ( r-HuEPO ) ameliorates this anemia , reducing the need for subsequent packed red blood cell ( PRBC ) transfusions .', ARRAY("paclitaxel", "carboplatin", "erythropoietin"), ARRAY("Paclitaxel", "Carboplatin", "Erythropoietin", "Recombinant", "human", "erythropoietin")),
('Combination therapy with fluconazole and flucytosine in the murine model of cryptococcal meningitis .', ARRAY("fluconazole", "flucytosine"), ARRAY("Fluconazole", "Flucytosine")),
('The consumption of meropenem or doripenem was calculated using the Anatomic Therapeutic Chemical classification and defined daily doses methodology .', ARRAY("meropenem", "doripenem"), ARRAY("Meropenem", "doripenem")),
('After successful phase II studies , recent phase III trials established combinations of chlorambucil with anti-CD20 antibodies such as rituximab , ofatumumab and obinutuzumab as a valuable treatment option for these patients .', ARRAY("chlorambucil", "rituximab", "ofatumumab", "obinutuzumab"), ARRAY("Chlorambucil", "Rituximab", "Obinutuzumab", "Ofatumumab")),
('Unorthodox antibiotic combinations including ciprofloxacin against high-level gentamicin resistant enterococci .', ARRAY("ciprofloxacin", "gentamicin"), ARRAY()),
('Randomized Trial of Lenalidomide Alone Versus Lenalidomide Plus Rituximab in Patients With Recurrent Follicular Lymphoma : CALGB 50401 ( Alliance ) .', ARRAY("Lenalidomide", "Lenalidomide", "Rituximab"), ARRAY("Lenalidomide", "Rituximab")),
('Potential cost-effectiveness of rifampin vs. isoniazid for latent tuberculosis : implications for future clinical trials .', ARRAY("rifampin", "isoniazid"), ARRAY()),
('Treatments included once daily erlotinib , which was given alone for the first 7 days of treatments , then in combination with once daily sirolimus .', ARRAY("erlotinib", "sirolimus"), ARRAY()),
('Cabazitaxel effectively killed PC-3R cells , and MDR1 knockdown improved the sensitivity of PC-3R cells to docetaxel but not to cabazitaxel .', ARRAY("Cabazitaxel", "docetaxel", "cabazitaxel"), ARRAY("Cabazitaxel", "Docetaxel")),
('The results suggest that rational therapy for severe CHF includes addition of the aldosterone antagonist spironolactone to low doses of captopril ( or another ACE inhibitor ) and high doses of loop diuretics , provided renal function is adequate .', ARRAY("spironolactone", "captopril", "loop", "diuretics"), ARRAY("Spironolactone", "Captopril")),
('The MICs of erythromycin and clindamycin for most of the LAB were within the normal range of susceptibility .', ARRAY("erythromycin", "clindamycin"), ARRAY("Clindamycin", "Erythromycin")),
('Based on HRs for RFS/DFS , the risk of recurrence with nivolumab was similar to that of pembrolizumab and lower than that of ipilimumab 3 mg/kg , ipilimumab 10 mg/kg , or interferon .', ARRAY("nivolumab", "pembrolizumab", "ipilimumab", "ipilimumab"), ARRAY("nivolumab", "pembrolizumab", "ipilimumab", "ipilimumab", "interferon")),
('Combination therapy of infliximab and methotrexate is more effective in reducing clinical and biochemical disease activity than gold with methylprednisolone treatment in RA patients during 22 weeks of treatment , especially in the first 6 weeks .', ARRAY("infliximab", "methotrexate", "methylprednisolone", "gold"), ARRAY("Infliximab", "Methotrexate")),
('Serum homocysteine , cholesterol , retinol , alpha-tocopherol , glycosylated hemoglobin and inflammatory response during therapy with bevacizumab , oxaliplatin , 5-fluorouracil and leucovorin .', ARRAY("bevacizumab", "oxaliplatin", "5-fluorouracil", "leucovorin"), ARRAY("Serum", "homocysteine", "cholesterol", "retinol", "alpha-tocopherol", "glycosylated", "hemoglobin", "inflammatory", "response", "bevacizumab", "oxaliplatin", "5-fluorouracil", "leucovorin")),
('Reversal of drug resistance by planetary ball milled ( PBM ) nanoparticle loaded with resveratrol and docetaxel in prostate cancer .', ARRAY("resveratrol", "docetaxel"), ARRAY("Reversal", "of", "drug", "resistance", "by", "planetary", "ball", "milled", "(", "PBM", ")", "nanoparticle", "loaded", "with", "resveratrol", "and", "docetaxel", "in", "prostate", "cancer")),
('Records of 63 patients diagnosis of IHCC who received Gemcitabine and Carboplatin ( G-C Regimen ) chemotherapy as a first line were retrospectively reviewed .', ARRAY("Gemcitabine", "Carboplatin"), ARRAY("Gemcitabine", "Carboplatin")),
('Drugs most commonly implicated in ADRs were amoxicillin + clavulanate ( 21.87 % ) followed by ceftriaxone ( 20.31 % ) .', ARRAY("amoxicillin", "clavulanate", "ceftriaxone"), ARRAY("Amoxicillin", "clavulanate", "ceftriaxone")),
('Dexamethasone and piroxicam provided in the diet were found to significantly inhibit lung tumors induced by 60 mg/kg vinyl carbamate at 24 weeks whereas myo-inositol also provided in the diet , did not significantly inhibit tumor formation .', ARRAY("Dexamethasone", "piroxicam"), ARRAY("Diethylstilbestrol", "Dexamethasone", "Piroxicam", "Myo-inositol")),
('We identified 938 patients with RCC who had initially been treated with sunitinib ( n = 554 ) or sorafenib ( n = 384 ) .', ARRAY("sunitinib", "sorafenib"), ARRAY("Renal", "Cell", "Carcinoma", "Sunitinib", "Sorafenib")),
('( 3 ) Observed facilitatory effects of caerulein on the hypothalamic defensive attack were very similar to those observed with dopamine ( DA ) agonists such as methamphetamine and apomorphine and opposite to those with DA antagonists such as haloperidol and chlorpromazine .', ARRAY("methamphetamine", "haloperidol", "chlorpromazine"), ARRAY("Apomorphine", "Methamphetamine", "Haloperidol", "Chlorpromazine", "Dopamine")),
('The NIBIT-MESO-1 study demonstrated the efficacy and safety of tremelimumab combined with durvalumab in patient with unresectable mesothelioma followed up for a median of 52 months [ IQR 49 - 53 ] .', ARRAY("tremelimumab", "durvalumab"), ARRAY("NIBIT-MESO-1", "tremelimumab", "durvalumab")),
('Sorafenib has been the standard of care for a decade , and promising results for regorafenib as a second-line and lenvatinib as a first-line treatment were reported only 1 or 2 years ago .', ARRAY("Sorafenib", "regorafenib", "lenvatinib"), ARRAY("Regorafenib", "Sorafenib")),
('The addition of trastuzumab , pertuzumab , bevacizumab , or lapatinib to chemotherapy significantly ( P < .05 ) improved objective response rate ( ORR ) , time to failure ( TTF ) , and overall survival ( OS ) in patients with HER2-positive ( HER2(+ ) ) disease .', ARRAY("trastuzumab", "pertuzumab", "bevacizumab", "lapatinib", "chemotherapy"), ARRAY("Trastuzumab", "Pertuzumab", "Bevacizumab", "Lapatinib")),
('The feasibility and efficacy of an intensified procarbazine-free consolidation regimen VECOPA ( vinblastine , etoposide , cyclophosphamide , vincristine , prednisone , doxorubicin ) were investigated .', ARRAY("vinblastine", "etoposide", "cyclophosphamide", "vincristine", "prednisone", "doxorubicin"), ARRAY("vinblastine", "etoposide", "cyclophosphamide", "vincristine", "prednisone", "doxorubicin")),
('The combination regimen of nivolumab plus ipilimumab demonstrates activity in metastatic uveal melanoma , with deep and sustained confirmed responses .', ARRAY("nivolumab", "ipilimumab"), ARRAY("nivolumab", "ipilimumab")),
('The FDA reviewed data in electronic format from a randomized controlled clinical trial of 1106 adult patients with newly diagnosed Philadelphia chromosome-positive CML in chronic phase , comparing imatinib with the combination of IFN-alpha and cytarabine .', ARRAY("imatinib", "cytarabine", "IFN-alpha"), ARRAY("Imatinib", "Gleevec", "Dasatinib", "BCR-ABL", "Cytarabine", "IFN-alpha")),
('A total of 164 patients with recurrent ovarian cancer were selected and randomly divided into two groups : experimental group ( n=82 , BEV + paclitaxel + carboplatin ) and control group ( n=82 , paclitaxel + carboplatin ) .', ARRAY("paclitaxel", "carboplatin", "paclitaxel", "carboplatin", "BEV"), ARRAY()),
('Successful treatment of an adult with Kasabach-Merritt syndrome using thalidomide , vincristine , and prednisone .', ARRAY("thalidomide", "vincristine", "prednisone"), ARRAY("Thalidomide", "Vincristine", "Prednisone")),
('Superiority of sirolimus ( rapamycin ) over cyclosporine in augmenting allograft and xenograft survival in mice treated with antilymphocyte serum and donor-specific bone marrow .', ARRAY("sirolimus", "rapamycin", "cyclosporine"), ARRAY("rapamycin", "cyclosporine")),
('Profiling of drug-metabolizing enzymes/transporters in CD33 + acute myeloid leukemia patients treated with Gemtuzumab-Ozogamicin and Fludarabine , Cytarabine and Idarubicin .', ARRAY("Fludarabine", "Cytarabine", "Idarubicin", "Gemtuzumab-Ozogamicin"), ARRAY("Gemtuzumab", "Ozogamicin", "Fludarabine", "Cytarabine", "Idarubicin")),
('In addition , anti-PCSK9 drugs ( evolocumab and alirocumab ) provide an effective solution for patients with familial hypercholesterolemia ( FH ) and statin intolerance at very high cardiovascular risk .', ARRAY("evolocumab", "alirocumab"), ARRAY("Alirocumab", "Evolocumab")),
('Likewise , emotional , functional well-being , and QoL aspects specifically related to lung cancer were better improved in the Gefitinib group than in the Pemetrexed group ( P<0.05 ) .', ARRAY("Gefitinib", "Pemetrexed"), ARRAY("Gefitinib", "Pemetrexed")),
('To determine efficacy and safety of the association of IL-1 receptor antagonist anakinra plus methylprednisolone in severe COVID-19 pneumonia with hyperinflammation .', ARRAY("anakinra", "methylprednisolone"), ARRAY()),
('When rats were given Li for 3 days , followed by a single injection of imipramine , the concentrations of desipramine in the brain and serum were higher than those in the vehicle-treated rat , although the imipramine concentrations in both tissues did not differ in Li- and in vehicle-treated rats .', ARRAY("Li", "imipramine", "desipramine", "imipramine"), ARRAY("Desipramine", "Imipramine")),
('Using population-based data from the province of Ontario , the benefit of first-line ipilimumab was estimated by comparing outcomes of patients treated with first-line dacarbazine over the period 2007 - 2009 with patients treated over the period 2010 - 2015 with first-line ipilimumab .', ARRAY("ipilimumab", "dacarbazine", "ipilimumab"), ARRAY("Ipilimumab", "Dacarbazine")),
('In the present study the combination of tamoxifen and bromocriptine was tried for the suppression of prolactin in prolactin secreting adenomas which were resistant to suppression with bromoergocriptine alone .', ARRAY("tamoxifen", "bromocriptine"), ARRAY("tamoxifen", "bromocriptine", "bromoergocriptine")),
('The present case suggests that combination therapy with thalidomide and dexamethasone is still an alternative treatment regimen for resistant extramedullary plasmacytoma with a plasmablastic morphology .', ARRAY("thalidomide", "dexamethasone"), ARRAY("Thalidomide", "Dexamethasone")),
('Among 788 eligible patients , the median ( SD ) age was 59 ( 22 - 74 ) years ; 263 patients were assigned to doxorubicin plus cisplatin treatment , 263 patients to docetaxel plus cisplatin treatment , and 262 patients to paclitaxel plus carboplatin treatment .', ARRAY("doxorubicin", "cisplatin", "docetaxel", "cisplatin", "paclitaxel", "carboplatin"), ARRAY("doxorubicin", "docetaxel", "paclitaxel", "cisplatin"));
"""

# COMMAND ----------

if not spark.catalog.tableExists(f"main.fine_tuning_workshop.baseline_eval_results"):
  spark.sql("DROP TABLE IF EXISTS main.fine_tuning_workshop.baseline_eval_results;")
  
  spark.sql("""CREATE TABLE IF NOT EXISTS main.fine_tuning_workshop.baseline_eval_results (
    sentence	STRING,
    human_annotated_entities	ARRAY<STRING>,
    baseline_predictions_cleaned	ARRAY<STRING>
    );""")
  
  spark.sql(insert_statement)