import os
import pandas as pd

if __name__ == '__main__':
    # Test set
    df = pd.read_parquet('/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/integration/integration_test.parquet')
    print(df["data_source"])
    df['data_source'] = 'integration_numeric'
    print(df["data_source"])
    df.to_parquet('/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/integration_3b_llmjudge/integration_test.parquet')

    # Train set
    df = pd.read_parquet('/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/integration/integration_train.parquet')
    print(df["data_source"])
    df['data_source'] = 'llm_judge_integration'
    print(df["data_source"])
    df.to_parquet('/home/ubuntu/o1-replication-usmid/CustomTinyZero/data/integration_3b_llmjudge/integration_train.parquet')