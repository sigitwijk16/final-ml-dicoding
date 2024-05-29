"""Transform module
"""

import tensorflow as tf
import tensorflow_transform as tft

# Define categorical features and their vocab sizes
CATEGORICAL_FEATURES = {
    "generalAppearance": 4,  # Rating scale 2-5
    "mannerOfSpeaking": 4,  # Rating scale 2-5
    "physicalCondition": 4,  # Rating scale 2-5
    "mentalAlertness": 4,  # Rating scale 2-5
    "selfConfidence": 4,  # Rating scale 2-5
    "abilityToPresentIdeas": 4,  # Rating scale 2-5
    "communicationSkills": 4,  # Rating scale 2-5
    "studentPerformanceRating": 4,  # Rating scale 2-5
}

# Define label key
LABEL_KEY = "class"

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def convert_num_to_one_hot(label_tensor, num_labels):
    """
    Convert a label into a one-hot vector
    Args:
        label_tensor: Tensor containing labels
        num_labels: Number of unique labels
    
    Returns:
        One-hot encoded tensor
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: Map from feature keys to raw features.
    
    Returns:
        Map from feature keys to transformed features.    
    """
    
    outputs = {}
    
    # Processing categorical features
    for key, vocab_size in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(
            inputs[key], top_k=vocab_size, num_oov_buckets=1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=vocab_size + 1  # Include OOV bucket
        )
    
    # Convert label to int and one-hot encode it
    outputs[transformed_name(LABEL_KEY)] = convert_num_to_one_hot(tf.cast(inputs[LABEL_KEY], tf.int64), num_labels=2)  # Assuming two classes
    
    return outputs
