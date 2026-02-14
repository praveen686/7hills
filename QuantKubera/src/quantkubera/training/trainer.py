"""Trainer class for MomentumTransformer."""

import os
import tensorflow as tf
from tensorflow import keras
from typing import Optional, Dict, Any
from quantkubera.models.tft import MomentumTransformer
from quantkubera.models.losses import SharpeLoss


class MomentumTrainer:
    """Handles training of Momentum Transformer models.
    
    Args:
        model: MomentumTransformer instance
        learning_rate: Learning rate for optimizer
        model_dir: Directory to save checkpoints
    """
    
    def __init__(
        self,
        model: MomentumTransformer,
        learning_rate: float = 1e-3,
        model_dir: str = 'models/trained'
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.model_dir = model_dir
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Compile model with Sharpe Loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=SharpeLoss(),
            metrics=[SharpeLoss()]  # Track Sharpe as metric too
        )
        
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 100,
        patience: int = 10,
        verbose: int = 1,
        extra_callbacks: Optional[list] = None
    ) -> keras.callbacks.History:
        """Train the model.
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Verbosity level
            extra_callbacks: Additional Keras callbacks
            
        Returns:
            Training history
        """
        callbacks = self._create_callbacks(patience)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def _create_callbacks(self, patience: int) -> list:
        """Create training callbacks."""
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_path = os.path.join(self.model_dir, 'best_model.keras')
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',  # Lower Sharpe loss is better (negative Sharpe)
                verbose=1
            )
        )
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
        )
        
        # Reduce LR on plateau
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
        )
        
        # TensorBoard
        log_dir = os.path.join(self.model_dir, 'logs')
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq=10
            )
        )
        
        return callbacks
    
    def save_model(self, filepath: Optional[str] = None):
        """Save model to file."""
        if filepath is None:
            filepath = os.path.join(self.model_dir, 'final_model.keras')
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load model from file."""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'SharpeLoss': SharpeLoss}
        )
        print(f"Model loaded from {filepath}")
        return self.model
    
    def evaluate(self, test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """Evaluate model on test set."""
        results = self.model.evaluate(test_dataset, return_dict=True)
        return results
