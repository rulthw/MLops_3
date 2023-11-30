from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import datetime as dt

args = {
    "owner": "admin",
    "start_date": dt.datetime(2023, 11, 27),
    "depends_on_past": False
}

with DAG(
    dag_id='regression_score',
    default_args=args,
    schedule_interval=None,
    tags=['score']) as dag:
    get_data = BashOperator(task_id='get_data',
                            bash_command="python3 ~/MLops_3/scripts/get_data.py", 
                            dag=dag)
    process_data = BashOperator(task_id='process_data',
                            bash_command="python3 ~/MLops_3/scripts/process_data.py", 
                            dag=dag)
    train_test_split_data = BashOperator(task_id='train_test_split_data',
                            bash_command="python3 ~/MLops_3/scripts/train_test_split.py", 
                            dag=dag)  
    train_model = BashOperator(task_id='train_model',
                            bash_command="python3 ~/MLops_3/scripts/train_model.py", 
                            dag=dag)
    test_model = BashOperator(task_id='test_model',
                            bash_command="python3 ~/MLops_3/scripts/test_model.py", 
                            dag=dag)
    get_data >> process_data >> train_test_split_data >> train_model >> test_model
