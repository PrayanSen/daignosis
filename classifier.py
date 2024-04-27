import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import xgboost as xgb

from read_neo4j import neo_driver, extract_features


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
    res.to_csv("task_A.csv", index=False)


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
    res.to_csv("task_B.csv", index=False)


if __name__ == "__main__":

    # df = pd.read_csv("samples_with_disease_gene_phenotype.csv")
    query = """
        MATCH (sample:Biological_sample)-[:HAS_DISEASE]->(disease:Disease)
        OPTIONAL MATCH (sample)-[:HAS_DAMAGE]->(damage:Gene)
        OPTIONAL MATCH (sample)-[:HAS_PHENOTYPE]->(phenotype:Phenotype)
        OPTIONAL MATCH (damage)-[:TRANSCRIBED_INTO]->(transcript:Transcript)-[:LOCATED_IN]->(chromosome:Chromosome)
        RETURN sample, disease, damage, phenotype, chromosome
    """
            # MATCH (sample:Biological_sample)-[:HAS_DISEASE]->(disease:Disease)
            # OPTIONAL MATCH (sample)-[:HAS_DAMAGE]->(damage:Gene)
            # OPTIONAL MATCH (sample)-[:HAS_PHENOTYPE]->(phenotype:Phenotype)
            # RETURN sample, disease, damage, phenotype

    driver = neo_driver()
    df_A, df_B = extract_features(driver, query)

    run_task_A(df_A)
    run_task_B(df_B)


