# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-Tuning and Evaluation: Enhanced Named Entity Recognition (NER)
# MAGIC
# MAGIC In Lab 2, we’ll go beyond prompt engineering by fine-tuning the Llama 3.2 1b model for a medical Named Entity Recognition (NER) task, specifically for drug name extraction. This fine-tuning process will allow us to tailor the model to perform with greater accuracy and reliability on this specialized task.
# MAGIC
# MAGIC ### Lab Objectives
# MAGIC In this lab, you will:
# MAGIC 1. **Fine-Tune the Model**: We’ll apply instruction fine-tuning to adapt the model, training it on the inital dataset to identify drug names more effectively.
# MAGIC 2. **Evaluate Outputs**: Once fine-tuned, we’ll assess the model’s performance on the same Drug Name Extraction task.
# MAGIC 3. **Compare Results**: Finally, we’ll compare the fine-tuned model’s outputs to the baseline results from Lab 1, highlighting improvements in percision, recall, and processing time.
# MAGIC
# MAGIC By the end of this lab, you will gain hands-on experience in transforming a general-purpose language model into a specialized, high-performance tool, with tangible gains over prompt engineering alone.

# COMMAND ----------

# DBTITLE 1,Install Libraries
# Let's start by installing our dependencies
%pip install databricks-genai==1.0.2 langchain-community==0.2.0 transformers==4.31.0
%pip install databricks-sdk==0.27.1 mlflow

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Run setup script and load helper functions
# MAGIC %run ./_resources/0.0-env_setup

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM main.fine_tuning_workshop.baseline_eval_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Training and Test Dataset
# MAGIC The hugging face dataset from the previous lab is already split into training and test set. 

# COMMAND ----------

# DBTITLE 1,Load training and eval tables
training_df = spark.table("drug_extraction_train_data")
eval_df = spark.table("drug_extraction_eval_data")

print("Number of Training Records: " + str(training_df.count()))
print("Number of Test Records: " + str(eval_df.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building our Prompt
# MAGIC We will utilize the same prompt as the previous lab in the fine-tuning process. The system prompt will be crucial when we format our data for training

# COMMAND ----------

# DBTITLE 1,Initalize Prompt
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

# MAGIC %md-sandbox
# MAGIC ## Preparing the Dataset for Chat Completion
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/llm-fine-tuning/databricks-llm-fine-tuning-classif-2.png?raw=true" width="700px" style="float: right">
# MAGIC
# MAGIC Using the completion API is always recommended as default option as Databricks will properly format the final training prompt for you.
# MAGIC
# MAGIC Chat completion requires a list of **role** and **prompt**, following the OpenAI standard. This standard has the benefit of transforming our input into a prompt following our LLM instruction pattern. <br/>
# MAGIC Note that each foundation model might be trained with a different instruction type, so it's best to use the same type when fine tuning.<br/>
# MAGIC *We recommend using Chat Completion whenever possible.*
# MAGIC
# MAGIC ```
# MAGIC [
# MAGIC   {"role": "system", "content": "[system prompt]"},
# MAGIC   {"role": "user", "content": "Here is a documentation page:[RAG context]. Based on this, answer the following question: [user question]"},
# MAGIC   {"role": "assistant", "content": "[answer]"}
# MAGIC ]
# MAGIC ```
# MAGIC
# MAGIC *Remember that your Fine Tuning dataset should be the same format as the one you're using for your RAG application.<br/>*
# MAGIC
# MAGIC #### Training Data Type
# MAGIC
# MAGIC Databricks supports a large variety of dataset formats (Volume files, Delta tables, and public Hugging Face datasets in .jsonl format), but we recommend preparing the dataset as Delta tables within your Catalog as part of a proper data pipeline to ensure production quality.
# MAGIC *Remember, this step is critical and you need to make sure your training dataset is of high quality.*
# MAGIC
# MAGIC Let's create a small pandas UDF to help create our final chat completion dataset.<br/>

# COMMAND ----------

# DBTITLE 1,Format the data
from pyspark.sql.functions import pandas_udf, to_json
import pandas as pd

@pandas_udf("array<struct<role:string, content:string>>")
def create_conversation(sentence: pd.Series, entities: pd.Series) -> pd.Series:
    def build_message(s, e):
        # Default behavior with system prompt
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(s)},
            {"role": "assistant", "content": e}]
                
    # Apply build_message to each pair of sentence and entity
    return pd.Series([build_message(s, e) for s, e in zip(sentence, entities)])

# COMMAND ----------

# DBTITLE 1,format_data
from pyspark.sql.functions import pandas_udf, to_json

base_model_name = "meta_llama_v3_2_1b_instruct"

training_df = training_df.withColumn("human_annotated_entities", to_json("human_annotated_entities"))
training_df.select(create_conversation("sentence", "human_annotated_entities").alias('messages')).write.mode('overwrite').saveAsTable("chat_completion_training_dataset")

eval_df = eval_df.withColumn("human_annotated_entities", to_json("human_annotated_entities"))
eval_df.select(create_conversation("sentence", "human_annotated_entities").alias('messages')).write.mode('overwrite').saveAsTable("chat_completion_eval_dataset")

display(spark.table("chat_completion_eval_dataset"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Starting a Fine Tuning Run
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/llm-fine-tuning/databricks-llm-fine-tuning-classif-3.png?raw=true" width="700px" style="float: right">
# MAGIC
# MAGIC Once the training is done, your model will automatically be saved within Unity Catalog and available for you to serve!
# MAGIC
# MAGIC In this demo, we'll be using the API on the table we just created to programatically fine tune our LLM.
# MAGIC
# MAGIC However, you can also create a new Fine Tuning experiment from the UI!

# COMMAND ----------

# DBTITLE 1,Start Fine Tuning
from databricks.model_training import foundation_model as fm

model_to_train = "meta-llama/Llama-3.2-1B-Instruct"
registered_model_name = f"{catalog}.{db}.drug_extraction_ft_" + re.sub(r'[^a-zA-Z0-9]', '_',  base_model_name.lower())
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("user").get()

run_20ep = fm.create(
  data_prep_cluster_id = get_current_cluster_id(),  # Required if you are using delta tables as training data source. This is the cluster id that we want to use for our data prep job. See ./_resources for more details
  model=model_to_train,
  experiment_path=f"/Users/{username}/baseline-finetune-comparison-1b",
  train_data_path=f"{catalog}.{db}.chat_completion_training_dataset",
  eval_data_path=f"{catalog}.{db}.chat_completion_eval_dataset",
  task_type = "CHAT_COMPLETION",
  register_to=registered_model_name,
  training_duration='20ep' # Duration of the finetuning run, Check the training run metrics to know when to stop it (when it reaches a plateau)
)

run_20ep

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE:** You can run several fine-tuning jobs simultenaously to test different sets of hyperparameters.

# COMMAND ----------

# DBTITLE 1,Track Progress
displayHTML(f'Open the <a href="/ml/experiments/{run_20ep.experiment_id}/runs/{run_20ep.run_id}/model-metrics">training run on MLflow</a> to track the metrics')
display(run_20ep.get_events())

# Helper function waiting on the run to finish - see the _resources folder for more details
wait_for_run_to_finish(run_20ep)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Deploy the Fine-Tuned Model
# MAGIC With our model saved to Unity Catalog, we can deploy it with Databricks model serving. Once deployed, we can perform batch inference and see how it compares to our initial baseline.

# COMMAND ----------

# DBTITLE 1,Launch Fine Tune Model Endpoint

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput, AutoCaptureConfigInput

serving_endpoint_name = "ft_workshop_finetuned_llm-1b"
w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_entities=[
        ServedEntityInput(
            entity_name=registered_model_name,
            entity_version=get_latest_model_version(registered_model_name),
            min_provisioned_throughput=0, # The minimum tokens per second that the endpoint can scale down to.
            max_provisioned_throughput=22000,# The maximum tokens per second that the endpoint can scale up to.
            scale_to_zero_enabled=True
        )
    ]
)

force_update = False #Set this to True to release a newer version (the demo won't update the endpoint to a newer model version by default)
existing_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)
if existing_endpoint == None:
    print(f"Creating the endpoint {serving_endpoint_name}, this will take a few minutes to package and deploy the endpoint...")
    w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
else:
  print(f"endpoint {serving_endpoint_name} already exist...")
  if force_update:
    w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=serving_endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC # Perform Batch Inference
# MAGIC Once again we will use AI Query to perform batch inference at scale. This time, we will leverage our fine-tuned model `ft_workshop_finetuned_llm` that was deployed with the previous step.

# COMMAND ----------

# DBTITLE 1,Batch Inference + Display Initial Results
sql_statement = f"""
  SELECT
    sentence,
    human_annotated_entities,
    baseline_predictions_cleaned,
    ai_query(
      'ft_workshop_finetuned_llm-1b',                -- Placeholder for the endpoint name
      CONCAT('{system_prompt}', sentence)            -- Placeholder for the prompt and input
    ) AS finetuned_predictions                       -- Placeholder for the output column
  FROM {catalog}.{schema}.baseline_eval_results      -- Placeholder for the table name
"""

finetuned_prediction_df = spark.sql(sql_statement)
finetuned_prediction_df.write.mode("overwrite").saveAsTable("main.fine_tuning_workshop.finetune_temp")

display(finetuned_prediction_df)

# COMMAND ----------

# DBTITLE 1,Clean the Records
finetuned_prediction_df  = spark.read.table("main.fine_tuning_workshop.finetune_temp")
finetuned_prediction_cleaned = clean_finetuned_results(finetuned_prediction_df)

display(finetuned_prediction_cleaned)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the experiment results and compare metrics
# MAGIC We will log the percision and recall to MLflow for comparison. With metrics captured for both our baseline and fine-tuned model, we can compare them to see how much we improved a simple light-weight llm. 

# COMMAND ----------

# DBTITLE 1,Logging Metrics in MLFlow Experiment Run
import mlflow

# Capture model metrics
precision, recall = calculate_metrics(finetuned_prediction_cleaned, 'finetuned_predictions_cleaned', 'human_annotated_entities')

print("Percision: ", precision)
print("Recall: ",  recall)

# Set Experiment
mlflow.set_experiment(run_20ep.experiment_path)

# Hardcoded metrics for now. Can get percision from a function and pass it here
with mlflow.start_run(run_id=run_20ep.run_id):
    mlflow.log_metrics({"precision": precision, "recall": recall})

# COMMAND ----------

# DBTITLE 1,Comparing Results of Baseline LLM to Fine-tuned LLM
plot_comparison(run_20ep.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC We have successfully fine-tuned a Drug Name Extraction Model, achieving significant improvements over the baseline model’s performance. This process has demonstrated the value of tailoring an open-source model to a specialized task, resulting in tangible gains.
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC 1. **Enhanced Results**: The fine-tuned model showed notable improvements in both precision and recall, yielding more accurate and reliable extractions.
# MAGIC 2. **Improved Performance**: Fine-tuning not only improved accuracy but also sped up the processing time, allowing the model to handle records faster than the baseline.
# MAGIC 3. **Cost Efficiency**: By operating a lightweight, open-source LLM, we reduce costs compared to closed-source models. Additionally, the model’s increased efficiency means that endpoints can be spun down sooner, further saving on operational expenses.
# MAGIC
# MAGIC With our fine-tuned model now hosted on a real-time endpoint, you’re ready to query it live and see its enhanced performance in action!
# MAGIC