import json
import logging
import os
import sys
from aoa import (
    aoa_create_context,
    ModelContext,
    record_evaluation_stats,
    save_plot,
    store_byom_tmp
)
from aoa.util.connections import execute_sql
from sklearn import metrics
from teradataml import (
    ClassificationEvaluator,
    configure,
    DataFrame,
    DataikuPredict,
    DataRobotPredict,
    get_context,
    H2OPredict,
    ONNXPredict,
    PMMLPredict,
    remove_context
)
from teradataml.analytics.valib import *


predictions_temp_table = f"preds_{str(uuid.uuid4()).replace('-','')}"
byom_temp_table = f"byom_{str(uuid.uuid4()).replace('-','')}"
safe_dir = os.getenv('PWD', '/opt/modelops/data')


def is_safe_path(file_path, raise_error=True):
    unsafe_dirs = []
    if os.path.commonprefix([os.path.realpath(file_path), safe_dir]) == safe_dir:
        return file_path
    else:
        unsafe_dirs.append(safe_dir)

    if raise_error:
        raise ValueError(f"Wrong path for the file, must be under {safe_dir}")

    return False


def evaluate_teradata_byom(model_type: str, context: ModelContext) -> DataFrame:
    configure.byom_install_location = os.environ.get("AOA_BYOM_INSTALL_DB", "MLDB")

    with open("metadata.json") as f:
        metadata = json.load(f)

    column_expression = metadata["trainedModel"]["metadata"]["byomColumnExpression"]

    with open(is_safe_path(f"{safe_dir}/{context.artifact_input_path}/model.{model_type.lower()}"), "rb") as f:
        model_bytes = f.read()

    model = store_byom_tmp(get_context(), byom_temp_table, context.model_version, model_bytes)

    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    test_df = DataFrame.from_query(context.dataset_info.sql)

    if model_type == "PMML":
        pmml = PMMLPredict(
            modeldata=model,
            newdata=test_df,
            accumulate=[entity_key, target_name])
        predictions_df = pmml.result

    elif model_type == "H2O":
        h2o = H2OPredict(
            modeldata=model,
            newdata=test_df,
            model_type="OpenSource",
            accumulate=[entity_key, target_name])
        predictions_df = h2o.result

    elif model_type == "H2O_DAI":
        license_file = f"{context.artifact_input_path}/license.txt"
        if not os.path.exists(license_file):
            logging.error("Missing license.txt file for H2O DAI model")
            sys.exit(2)
        with open(license_file,"r") as f:
            license = f.read()
        dai_model=model.assign(license=license)
        h2o = H2OPredict(
            modeldata=dai_model,
            newdata=test_df,
            model_type="DAI",
            accumulate=[entity_key, target_name])
        predictions_df = h2o.result

    elif model_type == "ONNX":
        onnx = ONNXPredict(
            modeldata=model,
            newdata=test_df,
            accumulate=[entity_key, target_name])
        predictions_df = onnx.result

    elif model_type == "DATAIKU":
        dataiku = DataikuPredict(
            modeldata=model,
            newdata=test_df,
            accumulate=[entity_key, target_name])
        predictions_df = dataiku.result

    elif model_type == "DATAROBOT":
        datarobot = DataRobotPredict(
            modeldata=model,
            newdata=test_df,
            accumulate=[entity_key, target_name])
        predictions_df = datarobot.result
    # workaround to make sure the df is available as a view
    predictions_df._DataFrame__execute_node_and_set_table_name(predictions_df._nodeid, predictions_df._metaexpr)

    predictions_df = DataFrame.from_query(f"""
    SELECT 
        {entity_key},
        {target_name} as y_test,
        {column_expression} AS {target_name}
    FROM {predictions_df._table_name}
    """)

    predictions_df.to_sql(table_name=predictions_temp_table, if_exists="replace", temporary=True)

    return DataFrame.from_query(f"SELECT * FROM {predictions_temp_table}")


def evaluate_teradata_sas(context: ModelContext) -> DataFrame:
    sasdb = os.environ.get("AOA_SAS_INSTALL_DB")

    with open("metadata.json") as f:
        metadata = json.load(f)

    sas_models_table = metadata["trainedModel"]["metadata"]["sas"]["table"]
    sas_models_db = metadata["trainedModel"]["metadata"]["sas"]["database"]
    column_expression = metadata["trainedModel"]["metadata"]["byomColumnExpression"]

    external_id = metadata["trainedModel"]["externalId"]

    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key
    # SAS_SCORE_EP procedure INQUERY parameter doesn't accept new lines.
    inquery = context.dataset_info.sql.replace("\n", " ")

    query = f"""
            CALL {sasdb}.SAS_SCORE_EP
            (
               'INQUERY={inquery}',
               'MODELTABLE={sas_models_db}.{sas_models_table}',
               'MODELNAME={external_id}',
               'OUTTABLE={predictions_temp_table}',
               'OUTKEY={entity_key}',
               'OPTIONS=VOLATILE=YES;'
            );
        """
    logging.debug(query)

    execute_sql(query)

    return DataFrame.from_query(f"""
        SELECT 
            {entity_key},
            {target_name} as y_test,
            {column_expression} AS {target_name}
        FROM {predictions_temp_table}
        """)


def evaluate(context: ModelContext, **kwargs):
    aoa_create_context()

    model_type = os.environ["MODEL_LANGUAGE"]

    if model_type in ["PMML", "H2O", "H2O_DAI", "ONNX", "DATAIKU", "DATAROBOT"]:
        predictions_df = evaluate_teradata_byom(model_type, context)

    elif model_type == "SAS":
        predictions_df = evaluate_teradata_sas(context)

    else:
        raise ValueError(f"Model type {model_type} not supported.")

    metrics_df = predictions_df.to_pandas()

    target_name = context.dataset_info.target_names[0]

    y_pred = metrics_df[[target_name]]
    y_test = metrics_df[["y_test"]]

    eval_df = DataFrame.from_query(f"""
        SELECT 
            Z.{context.dataset_info.entity_key}, Z.{context.dataset_info.target_names[0]} as Observed, Y.{context.dataset_info.target_names[0]} as Predicted
            FROM ({context.dataset_info.sql}) Z 
            LEFT JOIN (SELECT * FROM {predictions_temp_table}) Y ON Z.{context.dataset_info.entity_key} = Y.{context.dataset_info.entity_key}
        """)

    configure.val_install_location = os.environ.get("AOA_VAL_INSTALL_DB", os.environ.get("VMO_VAL_INSTALL_DB", "VAL")) # TODO: remove AOA reference in future version.
    statistics = valib.Frequency(data=eval_df, columns='Observed')

    try:
        eval_stats = ClassificationEvaluator(data=eval_df, observation_column='Observed', prediction_column='Predicted', num_labels=int(statistics.result.count(True).to_pandas().count_xval.iloc[0]))
        eval_data = eval_stats.output_data.to_pandas().reset_index(drop=True)

        evaluation = {
            'Accuracy': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Accuracy')].MetricValue.item()),
            'Recall': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Macro-Recall')].MetricValue.item()),
            'Precision': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Macro-Precision')].MetricValue.item()),
            'f1-score': '{:.2f}'.format(eval_data[eval_data.Metric.str.startswith('Macro-F1')].MetricValue.item())
        }
    except:
        evaluation = {
            'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
            'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
            'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
            'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
        }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    # create confusion matrix plot
    cf = metrics.confusion_matrix(y_test, y_pred)
    cm = metrics.ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=statistics.result.select('xval').to_pandas().xval.to_list())
    cm.plot()
    save_plot('Confusion Matrix', context=context)

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(features_df=DataFrame.from_query(context.dataset_info.sql),
                                predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_temp_table}"),
                                context=context)
    else:
        logging.debug("data_stats.json not found. Skipping compute statistics.")

    remove_context()
