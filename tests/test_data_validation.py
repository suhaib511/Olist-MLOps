import great_expectations as gx


def test_great_expectations_validation():
    context = gx.get_context(mode="file")

    checkpoint = context.checkpoints.get(
        "olist_train_checkpoint"
    )

    result = checkpoint.run()

    assert result.success is True