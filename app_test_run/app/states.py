from FeatureCloud.app.engine.app import AppState, app_state, Role
import time
import os
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb

from neo4j import GraphDatabase, Query, Record
from neo4j.exceptions import ServiceUnavailable
from pandas import DataFrame

from utils import read_config,write_output

from FeatureCloud.app.engine.app import AppState, app_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = read_config()


def run_task_A(df):
    print("Performing Task A")
    data = df.drop(columns=["disease_name", "disease_synonyms"])
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))

    data = pd.concat([data.drop(columns=categorical_cols), encoded_data], axis=1)

    labels = data[["subject_id", "class"]]

    train_x, test_x, train_y, test_y = train_test_split(data.drop(columns=["class"]), labels, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()
    # import pdb; pdb.set_trace()

    print("running model for A")

    model.fit(train_x.drop(columns=["subject_id"]), train_y["class"])

    print("predicting for A")

    pred_y = model.predict(test_x.drop(columns=["subject_id"]))

    f1 = f1_score(test_y["class"], pred_y)

    print("for A f1 score: ", f1)
    print(classification_report(y_true=test_y["class"], y_pred=pred_y))

    res = {
        "subject_id": test_y["subject_id"],
        "disease": pred_y
    }

    res = pd.DataFrame(res)

    f = "task_A.csv"

    res.to_csv(f, index=False)

    print("Written Results for Task A in file: ",  f)


def run_task_B(df):
    print("Performing Task B")

    label_encoder_chromosomes = LabelEncoder()
    df['chromosome'] = label_encoder_chromosomes.fit_transform(df['chromosome'])

    data = df.drop(columns=["disease_name"])
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != "class"]
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))

    data = pd.concat([data.drop(columns=categorical_cols), encoded_data], axis=1)
    # import pdb; pdb.set_trace()

    labels = data[["subject_id", "class"]]

    train_x, test_x, train_y, test_y = train_test_split(data.drop(columns=["class"]), labels, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier()

    label_encoder_classes = LabelEncoder()
    train_y_encoded = label_encoder_classes.fit_transform(train_y["class"])
    # test_y_encoded = label_encoder_classes.transform(test_y["class"])

    print("running model for B")

    model.fit(train_x.drop(columns=["subject_id"]), train_y_encoded)# train_y["class"])

    print("predicting for B")

    # pred_y = model.predict(test_x.drop(columns=["subject_id"]))
    pred_y_encoded = model.predict(test_x.drop(columns=["subject_id"]))
    pred_y = label_encoder_classes.inverse_transform(pred_y_encoded)

    # f1 = f1_score(test_y["class"], pred_y) #, average="weighted")
    acc = accuracy_score(test_y["class"], pred_y)

    # print("f1 score: ", f1)
    print("for B accuracy: ", acc)
    print(classification_report(y_true=test_y["class"], y_pred=pred_y))


    res = {
        "subject_id": test_y["subject_id"],
        "disease": pred_y
    }

    res = pd.DataFrame(res)
    f = "task_B.csv"

    res.to_csv(f, index=False)

    print("Written Results for Task B in file: ", f)


def extract_features(results):
    # with driver.session() as session:
    #     result = session.run(query)
    print("Extracting features")
    df_list_a = []
    df_list_b = []
    for i, record in enumerate(results):
        sample = record["sample"]
        disease = record["disease"]
        damage = record["disease"]
        phenotype = record["phenotype"]

        obj_a = {
            "subject_id": int(sample["subjectid"]),
            "disease_name": disease["name"],
            "disease_synonyms": disease["synonyms"],
            "gene_id": damage["id"],
            "phenotype_name": phenotype["name"],
            # "chromosome": chromosome, # np.nan if chromosome == "nan" or chromosome == np.nan else chromosome,
            "class": 1
        }

        if disease["name"] == "control":
            obj_a["class"] = 0
            df_list_a.append(obj_a)
            del obj_a
            continue
        df_list_a.append(obj_a)
        del obj_a

        chromosome = str(record["chromosome"]["name"]) if record["chromosome"] else "nan"
        if chromosome == np.nan or "nan" in chromosome:
            continue
        # else:
        class2 = None
        # import pdb; pdb.set_trace()
        for e in disease["synonyms"]:
            if e.startswith("ICD10"):
                class2 = e.split(":")[1][0]
        if not class2:
            continue
        obj_b = {
            "subject_id": int(sample["subjectid"]),
            "disease_name": disease["name"],
            # "description": disease["description"],
            "gene_id": damage["id"],
            "phenotype_name": phenotype["name"],
            "chromosome": chromosome, # np.nan if chromosome == "nan" or chromosome == np.nan else chromosome,
            "class": class2
        }
        df_list_b.append(obj_b)
        del obj_b

            # print("sub_id: ", obj["subject_id"])

    df_a = pd.DataFrame(df_list_a)
    df_b = pd.DataFrame(df_list_b)
    print("Extracted features")
    return df_a, df_b



@app_state('initial')
class ExecuteState(AppState):

    def register(self):
        self.register_transition('terminal', Role.BOTH)

        
    def run(self):

        # Get Neo4j credentials from config
        neo4j_credentials = config.get("neo4j_credentials", {})
        NEO4J_URI = neo4j_credentials.get("NEO4J_URI", "")
        NEO4J_USERNAME = neo4j_credentials.get("NEO4J_USERNAME", "")
        NEO4J_PASSWORD = neo4j_credentials.get("NEO4J_PASSWORD", "")
        NEO4J_DB = neo4j_credentials.get("NEO4J_DB", "")
        logger.info(f"Neo4j Connect to {NEO4J_URI} using {NEO4J_USERNAME}")
        
        # Driver instantiation
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        df_A, df_B = None, None
        
        # Create a driver session with defined DB
        with driver.session(database=NEO4J_DB) as session:
                
            # Example Query to Count Nodes 
            # node_count_query = "MATCH (n) RETURN count(n)"
            # query = """
            #             MATCH (sample:Biological_sample)-[:HAS_DISEASE]->(disease:Disease)
            #             OPTIONAL MATCH (sample)-[:HAS_DAMAGE]->(damage:Gene)
            #             OPTIONAL MATCH (sample)-[:HAS_PHENOTYPE]->(phenotype:Phenotype)
            #             RETURN sample, disease, damage, phenotype
            #         """

            query = """
                MATCH (sample:Biological_sample)-[:HAS_DISEASE]->(disease:Disease)
                OPTIONAL MATCH (sample)-[:HAS_DAMAGE]->(damage:Gene)
                OPTIONAL MATCH (sample)-[:HAS_PHENOTYPE]->(phenotype:Phenotype)
                OPTIONAL MATCH (damage)-[:TRANSCRIBED_INTO]->(transcript:Transcript)-[:LOCATED_IN]->(chromosome:Chromosome)
                RETURN sample, disease, damage, phenotype, chromosome
            """
        #
        #     # Use .data() to access the results array
            results = session.run(query) #.data()
            logger.info(results)

            write_output(f"{results}")
            df_A, df_B = extract_features(results=results)
        # Close the driver connection
        driver.close()
        run_task_A(df_A)

        run_task_B(df_B)


        return 'terminal'



