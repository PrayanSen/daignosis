import numpy as np
from neo4j import GraphDatabase
import pandas as pd
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")


def get_biological_samples_with_disease(biological_samples_has_disease_query):
    df_list= []
    with driver.session() as session:
        result = session.run(biological_samples_has_disease_query)

        # unique_labels = [record["relationshipType"] for record in result] # label for node labels

        # Process the result
        cols = []
        df_list = []
        # import pdb; pdb.set_trace()
        for i, record in enumerate(result):
            # record["p"].nodes[0] ---> Biological_sample
            # record["p"].nodes[-1] ---> Disease

            b_sample = record["p"].nodes[0]
            disease = record["p"].nodes[-1]

            if not cols:
                cols.append("subject_id")
                cols.extend(list(disease.keys()))

            obj = {
                "subjectid": b_sample["subjectid"],
                "class": 1
            }
            obj.update(disease._properties)
            print("SubId: ", b_sample["subjectid"])

            obj["disease"] = obj.pop("name")
            if obj["disease"] == "control":
                obj["class"] = 0

            df_list.append(obj)
            del obj

        df = pd.DataFrame(df_list)

        df.to_csv("biological_samples_has_disease.csv", index=False)


def neo_driver():
    uri = "bolt://83.229.84.12:7687"
    user_name = "tumaiReadonly"
    password = "MAKEATHON2024"
    neo4j_db = "graph2.db"
    driver = GraphDatabase.driver(uri=uri, auth=(user_name, password), database=neo4j_db)

    return driver


def extract_features(driver, query):
    with driver.session() as session:
        result = session.run(query)

        # unique_labels = [record["relationshipType"] for record in result] # label for node labels

        # Process the result
        print("Extracting features")
        df_list_a = []
        df_list_b = []
        # import pdb; pdb.set_trace()
        for i, record in enumerate(result):
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


if __name__ == "__main__":

    # Example Cypher query to retrieve data
    query = """
        MATCH (sample:Biological_sample)-[:HAS_DISEASE]->(disease:Disease)
        OPTIONAL MATCH (sample)-[:HAS_DAMAGE]->(damage:Gene)
        OPTIONAL MATCH (sample)-[:HAS_PHENOTYPE]->(phenotype:Phenotype)
        RETURN sample, disease, damage, phenotype
    """

    query2 = """
        MATCH (sample:Biological_sample)-[:HAS_DISEASE]->(disease:Disease)
        OPTIONAL MATCH (sample)-[:HAS_DAMAGE]->(damage:Gene)
        OPTIONAL MATCH (sample)-[:HAS_PHENOTYPE]->(phenotype:Phenotype)
        OPTIONAL MATCH (damage)-[:TRANSCRIBED_INTO]->(transcript:Transcript)-[:LOCATED_IN]->(chromosome:Chromosome)
        RETURN sample, disease, damage, phenotype, chromosome
    """
    driver = neo_driver()

    df1, df2 = extract_features(driver, query2)

    df1.to_csv("samples_with_disease_gene_phenotype.csv", index=False)
    df2.to_csv("samples_with_icd10.csv", index=False)
    # print("Unique Labels:", unique_labels)

