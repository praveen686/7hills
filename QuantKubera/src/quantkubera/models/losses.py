import tensorflow as tf

class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1, **kwargs):
        self.output_size = output_size
        super().__init__(**kwargs)

    def call(self, y_true, weights):
        # In TMT, the model outputs 'weights' (positions) and we maximize Sharpe.
        # y_true are the actual returns.
        # 'weights' argument here corresponds to y_pred (the model output).
        # Note: TF Loss signature is (y_true, y_pred).
        
        y_pred = weights 
        captured_returns = y_pred * y_true
        mean_returns = tf.reduce_mean(captured_returns)
        
        # Negative Sharpe Ratio (since we want to minimize loss)
        return -(
            mean_returns
            / tf.sqrt(
                tf.reduce_mean(tf.square(captured_returns))
                - tf.square(mean_returns)
                + 1e-9
            )
            * tf.sqrt(252.0)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size})
        return config
