from aoa import (
    record_training_stats,
    get_feature_stats_summary,
    aoa_create_context,
    ModelContext
)
from teradataml import DataFrame


def compute_statistics(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    feature_summary = get_feature_stats_summary(context.dataset_info.get_feature_metadata_fqtn())
    categorical_features = [f for f in feature_names if feature_summary[f.lower()] == 'categorical']

    train_df = DataFrame.from_query(context.dataset_info.sql)

    record_training_stats(train_df,
                          features=feature_names,
                          targets=[target_name],
                          categorical=categorical_features + [target_name],
                          context=context)
