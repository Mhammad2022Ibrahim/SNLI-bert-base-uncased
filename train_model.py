from transformers import TFAutoModelForSequenceClassification, create_optimizer, PushToHubCallback
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf
from data_snli import checkpoint, tf_train_dataset, tf_validation_dataset

import os
from dotenv import load_dotenv

load_dotenv()

HUB_TOKEN = os.getenv("HUB_TOKEN")

# Load the pre-trained model for sequence classification
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

# model.compile(
#     optimizer="adam",
#     loss=SparseCategoricalCrossentropy(from_logits=True),
#     metrics=["accuracy"],
# )
# # Train the model
# print("Training the model...")
# model.fit(
#     tf_train_dataset,
#     validation_data=tf_validation_dataset,
# )

batch_size = 8
num_epochs = 5
num_train_steps = len(tf_train_dataset) * num_epochs
num_warmup_steps = int(0.1 * num_train_steps)

# Hugging Face optimizer with learning rate scheduling and weight decay
optimizer, lr_schedule = create_optimizer(
    init_lr=5e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    weight_decay_rate=0.01
)

# Loss function
loss = SparseCategoricalCrossentropy(from_logits=True)

# Compile model
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# Optional: Push the model to Hugging Face Hub
# model.push_to_hub("snli-bert-base-uncased", organization="your_org_name", private=True, commit_message="Initial commit")
push_to_hub_callback = PushToHubCallback(output_dir="./snli-bert-base-uncased", 
                                         hub_model_id="Mhammad2023/snli-bert-base-uncased",
                                         hub_token=HUB_TOKEN)

# Train the model
print("Training the model...")
model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=num_epochs,
    batch_size=batch_size,
    callbacks=[push_to_hub_callback]
)