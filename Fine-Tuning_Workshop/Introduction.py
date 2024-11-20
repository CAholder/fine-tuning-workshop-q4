# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop: Fine-Tuning Small Open-Source Language Models
# MAGIC
# MAGIC Welcome to this hands-on workshop focused on fine-tuning small open-source language models (LLMs). In recent years, large language models have become essential in various fields, powering everything from customer support chatbots to advanced research tools. However, many commercially available, closed-source LLMs present barriers due to high costs and limitations in customization and adaptability. This workshop will introduce you to the benefits and techniques of fine-tuning smaller, open-source models to meet specific needs, which can not only reduce operational costs but also potentially yield better performance on tailored tasks.
# MAGIC
# MAGIC ### Why Fine-Tune Open-Source Models?
# MAGIC
# MAGIC Open-source models bring several advantages:
# MAGIC 1. **Customizability**: With open access to the model, you can adapt its behavior to better match specific use cases or unique data domains. 
# MAGIC 2. **Cost Efficiency**: Smaller models require less computational power, which can significantly lower operational expenses without sacrificing performance.
# MAGIC 3. **Improved Relevance**: Fine-tuning allows us to train the model with data aligned to the target tasks, improving accuracy and relevance.
# MAGIC
# MAGIC While large, closed-source models offer impressive capabilities out of the box, their lack of customization and cost-efficiency can limit their usefulness. In contrast, fine-tuned smaller models, like the Llama 1b model we’ll explore, can be tailored to be responsive while being affordable at scale.
# MAGIC
# MAGIC ### Workshop Structure
# MAGIC
# MAGIC This workshop is split into two labs:
# MAGIC 1. **Lab 1**: Reviewing baseline results produced from a small, out of the box model that has only been prompt engineered.
# MAGIC 2. **Lab 2**: Hands-on portion which we will fine-tune an open-source model to our specific use case. We will then perform batch inference with the fine-tuned LLM.
# MAGIC
# MAGIC By the end, you will gain the skills to fine-tune small LLMs to your use case, making them effective, specialized tools for your specific needs.
# MAGIC
# MAGIC Let’s start by evaluating a model's baseline performance. Move on to **[Lab 1]($./0.1-base_model_performance)** to get started!