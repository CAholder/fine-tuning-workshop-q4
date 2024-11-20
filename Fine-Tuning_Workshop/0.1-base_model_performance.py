# Databricks notebook source
# MAGIC %md
# MAGIC # Baseline Model Evaluation: Named Entity Recognition (NER)
# MAGIC
# MAGIC In this lab, we’ll evaluate the baseline performance of the Llama 3.2 1b model on a Named Entity Recognition (NER) task, specifically focused on extracting drug names from text. Using prompt engineering, we’ll explore how well the model performs out-of-the-box, assessing its ability to identify medical entities without any task-specific fine-tuning.
# MAGIC
# MAGIC This baseline evaluation is essential for understanding:
# MAGIC 1. **Limitations of Prompt Engineering**: While prompt engineering can guide a model’s behavior, its effectiveness is often limited when dealing with specialized tasks like medical NER.
# MAGIC 2. **Performance Boundaries**: By examining the baseline model’s output, we’ll set a performance reference to compare against future fine-tuned results.
# MAGIC 3. **Fine-Tuning Justification**: Observing the baseline results helps illustrate why instruction fine-tuning can significantly improve performance for niche extraction tasks.
# MAGIC
# MAGIC > **Note**: You do not need to run this notebook as part of the lab. Running it will incur additional costs due to setting up a model-serving endpoint, which is not required for today’s workshop.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Library Installs
# Let's start by installing our dependencies
%pip install databricks-genai==1.0.2 langchain-community==0.2.0 transformers==4.31.0
%pip install databricks-sdk==0.27.1 mlflow

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run setup script and load helper functions
# MAGIC %run ./_resources/0.0-env_setup

# COMMAND ----------

# DBTITLE 1,Launch Baseline Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

llm_model = base_model_name = "llama_v3_2_1b_instruct"
serving_endpoint_baseline_name = "ft_workshop_baseline_llm_1b"
latest_version = get_latest_model_version(f"system.ai.{base_model_name}")

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_baseline_name,
    served_entities=[
        ServedEntityInput(
            entity_name=f"system.ai.{llm_model}", #Make sure you're using the same base model as the one you're fine-tuning on for relevant evaluation!
            entity_version=latest_version,
            min_provisioned_throughput=0, # The minimum tokens per second that the endpoint can scale down to.
            max_provisioned_throughput=22000,# The maximum tokens per second that the endpoint can scale up to.
            scale_to_zero_enabled=True
        )
    ]
)

existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_baseline_name), None
)
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_baseline_name}, this will take a few minutes to package and deploy the LLM...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_baseline_name, config=endpoint_config)
else:
  print(f"endpoint {serving_endpoint_baseline_name} already exist")

# COMMAND ----------

# MAGIC %md
# MAGIC # The Objective - Drug Name Extraction
# MAGIC We will utilize an LLM to extract the name of drugs from a given sentence. In traditional ETL, extracting these names would require complex logic. However, we can utilize an LLM to perform complex operations for both streaming and batch pipelines
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC - **Input:** We hypothesized that Aurora A kinase ( AK ) contributes to castrate resistance in prostate cancer ( PCa ) and that inhibiting AK with alisertib can resensitize PCa cells to androgen receptor ( AR ) inhibitor abiraterone . 
# MAGIC - **Output:** ["alisertib","abiraterone"]
# MAGIC
# MAGIC
# MAGIC ###The Dataset
# MAGIC We will be utilizing a Drug Combination Extraction found on [Hugging-Face](https://huggingface.co/datasets/allenai/drug-combo-extraction). The dataset will contain two columns:
# MAGIC 1. A sentence which contains one or many drugs
# MAGIC 2. A human annotated column containing the ground truth

# COMMAND ----------

# DBTITLE 1,Display Dataset
# MAGIC %sql
# MAGIC SELECT * FROM main.fine_tuning_workshop.drug_extraction_eval_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building our Prompt
# MAGIC We want to guide our LLM to extract the names of drugs from the provided sentences and place them into an array. To do so, we will utilize a prompt that provides guidelines on how to perform the task.

# COMMAND ----------

# DBTITLE 1,Define System Prompt
system_prompt = """
### INSTRUCTIONS:
You are a medical and pharmaceutical expert. Your task is to identify pharmaceutical drug names from the provided input and list them accurately. Follow these guidelines:

1. Do not add any commentary or repeat the instructions.
2. Extract the names of pharmaceutical drugs mentioned.
3. Place the extracted names in a Python list format. Ensure the names are enclosed in square brackets and separated by commas.
4. Maintain the order in which the drug names appear in the input.
5. Do not add any text before or after the list.
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extracting our entities with a baseline version (non fine-tuned)
# MAGIC
# MAGIC Before we fine tune a Named Entity Recogition LLM. Let's see how a baseline model performs with just prompt engineering.
# MAGIC
# MAGIC We will be using the endpoint `ft_workshop_baseline_llm` which launched earlier in this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Inference with AI_query
# MAGIC To easily perform Batch Inference with an LLM on Databricks, we can use [ai_query()](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html) to perform inference with simple sql. 

# COMMAND ----------

# DBTITLE 1,Batch Inference with ai_query
sql_statement = f"""
  SELECT
    sentence,
    human_annotated_entities,   
    ai_query(
      'ft_workshop_baseline_llm_1b',                       -- Placeholder for the endpoint name
      CONCAT('{system_prompt}', sentence)               -- Placeholder for the prompt and input column
    ) AS baseline_predictions                           -- Placeholder for the output column
  FROM {catalog}.{schema}.drug_extraction_eval_data     
"""

baseline_prediction_df = spark.sql(sql_statement)

# COMMAND ----------

# DBTITLE 1,Baseline LLM Outputs
display(baseline_prediction_df)

# COMMAND ----------

# DBTITLE 1,Clean the Results
baseline_prediction_cleaned = clean_baseline_prediction_results(baseline_prediction_df, 'baseline_predictions').drop("baseline_predictions")

# Show the result
display(baseline_prediction_cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculate Metrics
# MAGIC In use cases such as extraction and classification, we can calculate metrics such as precision and recall. We will use these metrics to compare the baseline results to the fine-tuned results

# COMMAND ----------

# DBTITLE 1,Calculate Performance Metrics
precision, recall = calculate_metrics(baseline_prediction_cleaned, 'baseline_predictions_cleaned', 'human_annotated_entities')

metrics = {'Precision': precision, 'Recall': recall}
plt.bar(metrics.keys(), metrics.values())
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Precision and Recall')

for i, (metric, value) in enumerate(metrics.items()):
    plt.text(i, value, f'{value:.2f}', ha='center', va='bottom')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Record the experiment
# MAGIC To log our results, we can utilize an MLflow experiment to record the metrics produced by our models.

# COMMAND ----------

# DBTITLE 1,Log an Experiment
import mlflow

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()
experiment_name = f"/Users/{username}/baseline-finetune-comparison-1b"
reset_experiment(experiment_name)

mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="baseline") as run:
    mlflow.log_param("model", llm_model)
    mlflow.log_metrics({"precision" : precision,
                        "recall" : recall
                        })
    mlflow.log_text(system_prompt, "system_prompt.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see how we can improve our results by Fine Tuning a model in [Lab 2]($./0.2-Fine-Tuning)